#!/usr/bin/env python
"""
Connects to robot, camera(s), gripper, and grasp server.
Runs grasps generated from grasp server.
"""

import time
import os
from os import system
import numpy as np
import sklearn
import matplotlib.pyplot as plt

import torch
import hydra
import omegaconf

from polygrasp.segmentation_rpc import SegmentationClient
from polygrasp.grasp_rpc import GraspClient
from polygrasp.serdes import load_bw_img

from polygrasp.robot_interface import GraspingRobotInterface
import graspnetAPI
import open3d as o3d
from typing import List
from scipy.spatial.transform import Rotation as R
from transforms3d.quaternions import mat2quat
import yaml

def compute_des_pose(best_grasp):
    """Convert between GraspNet coordinates to robot coordinates."""

    # Grasp point
    grasp_point = best_grasp.translation

    # Compute plane of rotation through three orthogonal vectors on plane of rotation
    grasp_approach_delta = best_grasp.rotation_matrix @ np.array([-0.3, 0.0, 0])
    grasp_approach_delta_plus = best_grasp.rotation_matrix @ np.array([-0.3, 0.1, 0])
    grasp_approach_delta_minus = best_grasp.rotation_matrix @ np.array([-0.3, -0.1, 0])
    bx = -grasp_approach_delta
    by = grasp_approach_delta_plus - grasp_approach_delta_minus
    bx = bx / np.linalg.norm(bx)
    by = by / np.linalg.norm(by)
    bz = np.cross(bx, by)
    plane_rot = R.from_matrix(np.vstack([bx, by, bz]).T)

    # Convert between GraspNet neutral orientation to robot neutral orientation
    des_ori = plane_rot * R.from_euler("y", 90, degrees=True)
    des_ori_quat = des_ori.as_quat()

    return grasp_point, grasp_approach_delta, des_ori_quat

def save_rgbd_masked(rgbd: np.ndarray, rgbd_masked: np.ndarray):
    num_cams = rgbd.shape[0]
    f, axarr = plt.subplots(2, num_cams)

    for i in range(num_cams):
        if num_cams > 1:
            ax1, ax2 = axarr[0, i], axarr[1, i]
        else:
            ax1, ax2 = axarr
        ax1.imshow(rgbd[i, :, :, :3].astype(np.uint8))
        ax2.imshow(rgbd_masked[i, :, :, :3].astype(np.uint8))

    f.savefig("rgbd_masked.png")
    plt.close(f)


def merge_pcds(pcds: List[o3d.geometry.PointCloud], eps=0.1, min_samples=2):
    """Cluster object pointclouds from different cameras based on centroid using DBSCAN; merge when within eps"""
    xys = np.array([pcd.get_center()[:2] for pcd in pcds])
    cluster_labels = (
        sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(xys).labels_
    )

    # Logging
    total_n_objs = len(xys)
    total_clusters = cluster_labels.max() + 1
    unclustered_objs = (cluster_labels < 0).sum()
    print(
        f"Clustering objects from all cameras: {total_clusters} clusters, plus"
        f" {unclustered_objs} non-clustered objects; went from {total_n_objs} to"
        f" {total_clusters + unclustered_objs} objects"
    )

    # Cluster label == -1 when unclustered, otherwise cluster label >=0
    final_pcds = []
    cluster_to_pcd = dict()
    for cluster_label, pcd in zip(cluster_labels, pcds):
        if cluster_label >= 0:
            if cluster_label not in cluster_to_pcd:
                cluster_to_pcd[cluster_label] = pcd
            else:
                cluster_to_pcd[cluster_label] += pcd
        else:
            final_pcds.append(pcd)

    return list(cluster_to_pcd.values()) + final_pcds

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def execute_grasp(
    robot: GraspingRobotInterface,
    chosen_grasp: graspnetAPI.Grasp,
    hori_offset: np.ndarray,
    time_to_go: float,
):
    """
    Executes a grasp. First attempts to grasp the robot; if successful,
    then the end-effector moves
        1. Up
        2. Horizontally by `hori_offset` (in meters),
        3. Down
        4. Releases the object by opening the gripper
    """
    traj, success = robot.grasp(
        chosen_grasp, time_to_go=time_to_go, gripper_width_success_threshold=0.001
    )
    print(f"Grasp success: {success}")

    if success:
        print("Placing object in hand to desired pose...")
        curr_pose, curr_ori = robot.get_ee_pose()
        print("Moving up")
        traj += robot.move_until_success(
            position=curr_pose + torch.Tensor([0, 0, 0.2]),
            orientation=curr_ori,
            time_to_go=time_to_go,
        )
        print("Moving horizontally")
        traj += robot.move_until_success(
            position=curr_pose + torch.Tensor([0, 0.0, 0.2]) + hori_offset,
            orientation=curr_ori,
            time_to_go=time_to_go,
        )
        print("Moving down")
        traj += robot.move_until_success(
            position=curr_pose + torch.Tensor([0, 0.0, 0.05]) + hori_offset,
            orientation=curr_ori,
            time_to_go=time_to_go,
        )

    print("Opening gripper")
    robot.gripper_open()
    curr_pose, curr_ori = robot.get_ee_pose()
    print("Moving up")
    traj += robot.move_until_success(
        position=curr_pose + torch.Tensor([0, 0.0, 0.2]),
        orientation=curr_ori,
        time_to_go=time_to_go,
    )

    return traj


@hydra.main(config_path="conf", config_name="run_grasp")
def main(cfg):
    print(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    print(f"Current working directory: {os.getcwd()}")

    print("Initialize robot & gripper")
    robot = hydra.utils.instantiate(cfg.robot)
    robot.k_grasp = 0.55    #0.55
    robot.k_approach = 0.9  #0.9
    robot.default_ee_quat = torch.Tensor([1, 0, 0, 0])
    # robot.default_ee_quat = torch.Tensor([0.00, 0.707, 0.00, 0.707])
    # robot.gripper_open()
    # robot.go_home()
    JOINT_GOAL = robot.get_joint_positions()
    JOINT_GOAL[0] = 0.56 #0.23
    JOINT_GOAL[1] = 0.08 #-0.15
    JOINT_GOAL[2] = 0.98 #1.23
    JOINT_GOAL[3] = -2.34 #-2.16
    JOINT_GOAL[4] = 0.06 #0.24
    JOINT_GOAL[5] = 1.33 #2.06
    JOINT_GOAL[6] = -0.95 #-1.84
    print("moving out of frame")
    robot.move_to_joint_positions(JOINT_GOAL)
    print("Initializing cameras")
    cfg.cam.intrinsics_file = hydra.utils.to_absolute_path(cfg.cam.intrinsics_file)
    cfg.cam.extrinsics_file = hydra.utils.to_absolute_path(cfg.cam.extrinsics_file)
    cameras = hydra.utils.instantiate(cfg.cam)

    print("Loading camera workspace masks")
    masks_1 = np.array(
        [load_bw_img(hydra.utils.to_absolute_path(x)) for x in cfg.masks_1],
        dtype=np.float64,
    )
    masks_2 = np.array(
        [load_bw_img(hydra.utils.to_absolute_path(x)) for x in cfg.masks_2],
        dtype=np.float64,
    )

    print("Connect to grasp candidate selection and pointcloud processor")
    system("cd ~/meta/fairo/perception/sandbox/polygrasp/ && export CUDA_HOME=usr/local/cuda && mrp down segmentation_server voxel grasp_server")
    time.sleep(1)
    system("cd ~/meta/fairo/perception/sandbox/polygrasp/ && export CUDA_HOME=usr/local/cuda && mrp up -f segmentation_server")
    time.sleep(2)
    segmentation_client = SegmentationClient()
    # grasp_client = GraspClient(
    #     view_json_path=hydra.utils.to_absolute_path(cfg.view_json_path)
    # )

    root_working_dir = os.getcwd()
    # for outer_i in range(cfg.num_bin_shifts):
    cam_i = 0 % 2
    print(f"=== Starting bin shift with cam {cam_i} ===")

    # Define some parameters for each workspace.
    if cam_i == 0:
        masks = masks_1
        hori_offset = torch.Tensor([0, -0.4, 0])
    else:
        masks = masks_2
        hori_offset = torch.Tensor([0, 0.4, 0])
    time_to_go = 10

    # for i in range(cfg.num_grasps_per_bin_shift):
    # Create directory for current grasp iteration
    os.chdir(root_working_dir)
    timestamp = int(time.time())
    os.makedirs(f"{timestamp}")
    os.chdir(f"{timestamp}")

    print(
        f"=== Grasp {1}/{cfg.num_grasps_per_bin_shift}, logging to"
        f" {os.getcwd()} ==="
    )

    print("Getting rgbd and pcds..")
    while(cameras.done != True):
        time.sleep(0.1)
        # print("Waiting")
    print("Done waiting")
    # cameras.sub = None
    rgbd = cameras.recent_rgbd
    # rgbd = cameras.get_rgbd()

    rgbd_masked = rgbd * masks[:, :, :, None]
    scene_pcd = cameras.get_pcd(rgbd)
    # try:
    #     save_rgbd_masked(rgbd, rgbd_masked)
    # except:
    #     pass

    print("Segmenting image...")
    unmerged_obj_pcds = []
    for i in range(cameras.n_cams):
        i=0 ##########################################
        force_cudnn_initialization()
        torch.cuda.empty_cache()
        try:
            obj_masked_rgbds, obj_masks = segmentation_client.segment_img(
                rgbd_masked[i], min_mask_size=cfg.min_mask_size
            )
        except:
            cameras.sub = None
            return
        unmerged_obj_pcds += [
            cameras.get_pcd_i(obj_masked_rgbd, i)
            for obj_masked_rgbd in obj_masked_rgbds
        ]
        break ######################################
    print(
        f"Merging {len(unmerged_obj_pcds)} object pcds by clustering their centroids"
    )
    obj_pcds = merge_pcds(unmerged_obj_pcds)
    if len(obj_pcds) == 0:
        print(
            f"Failed to find any objects with mask size > {cfg.min_mask_size}!"
        )
        # break
        cameras.sub = None
        return 
    # color_mask = np.array([0.48, 0.28, 0.28]) #tootpaste
    # color_mask = np.array([0.58, 0.37, 0.18]) #yellow
    # color_mask = np.array([0.12, 0.28, 0.19]) #green
    color_mask = np.array([0.49, 0.15, 0.14]) #red
    # color_mask = np.array([0.06, 0.21, 0.40]) #blue
    obj_i = 0 
    for obj_pcd in obj_pcds:
        points_color = np.asanyarray(obj_pcd.colors)
        avg_pt_color =np.mean(points_color,axis=1)
        filtered_points_color = points_color[(avg_pt_color<0.57)] #filter out white/table points
        avg_color = np.mean(filtered_points_color,axis=0)
        # avg_color = np.mean(points_color,axis=0)
        print(avg_color)
        o3d.visualization.draw_geometries([obj_pcd])
        check_array = np.abs(avg_color - color_mask) < 0.05 #0.09
        if(np.all(check_array)):
            print("got obj")
            filtered_obj_pcd = obj_pcd
            o3d.io.write_point_cloud("obj_pcd.ply", obj_pcd)
            o3d.visualization.draw_geometries([obj_pcd])
            break
        obj_i += 1
        if (obj_i == len(obj_pcds)):
            print("custom object not detected")
            return     
    filtered_obj_pcd = obj_pcds[obj_i]
    force_cudnn_initialization()
    torch.cuda.empty_cache()
    # from os import system
    system("cd ~/meta/fairo/perception/sandbox/polygrasp/ && export CUDA_HOME=usr/local/cuda && mrp down segmentation_server")
    time.sleep(5)
    system("cd ~/meta/fairo/perception/sandbox/polygrasp/ && export CUDA_HOME=usr/local/cuda && mrp up -f grasp_server")
    time.sleep(2)
    grasp_client = GraspClient(
        view_json_path=hydra.utils.to_absolute_path(cfg.view_json_path)
    )
    print("Getting grasps per object...")
    try:
        filtered_grasp_group = grasp_client.get_filtered_obj_grasps(filtered_obj_pcd, scene_pcd)
        # obj_i, filtered_grasp_group = grasp_client.get_obj_grasps(obj_pcds, scene_pcd)
    except:
        cameras.sub = None
        return

    print("Choosing a grasp for the object")
    # final_filtered_grasps, chosen_grasp_i, pre_grasp, approach, pre_arr, app_arr = robot.select_grasp(filtered_grasp_group, num_grasp_choices = 5)
    final_filtered_grasps, chosen_grasp_i, pre_grasp_arr, approach_arr, pre_grasp_jpos_arr, approach_jpos_arr = robot.select_grasp(filtered_grasp_group, num_grasp_choices = 5)
    if(len(final_filtered_grasps) == 0):
        print("no feasible grasps")
        cameras.sub = None
        return
    chosen_grasp = final_filtered_grasps[chosen_grasp_i]
    chosen_grasp_point, grasp_approach_delta, des_ori_quat = compute_des_pose(chosen_grasp)
    goal_pos = chosen_grasp_point + grasp_approach_delta * robot.k_grasp# chosen_grasp.translation
    pre_grasp = chosen_grasp_point + grasp_approach_delta * robot.k_approach
    goal_joint_pos = robot.ik(goal_pos, des_ori_quat)
    pre_grasp_joint_pos = robot.ik(pre_grasp, des_ori_quat)
    goal_mat = chosen_grasp.rotation_matrix
    goal_quat = des_ori_quat #mat2quat(goal_mat)
    goal_quat = np.array([goal_quat[3], goal_quat[0], goal_quat[1], goal_quat[2]])
    print(goal_pos)
    # print(pre_grasp_arr[:,chosen_grasp_i])
    # print(goal_mat)
    print(goal_quat)
    # print(final_filtered_grasps)
    print(pre_grasp)
    # print(chosen_grasp_point)
    # print(grasp_approach_delta)
    # print(robot.k_grasp)  
    # print(robot.k_approach)
    # print(app_arr)
    # print(chosen_grasp)

    

    # print("Going home")
    # robot.go_home()
    # obj_size = obj_pcds[obj_i].get_max_bound() - obj_pcds[obj_i].get_min_bound()
    dict_file = {"chosen_grasp_i": chosen_grasp_i+1, "gpos": goal_pos.tolist(), "pre_grasp": pre_grasp.tolist(), "pre_grasp_jpos": pre_grasp_joint_pos.tolist(), "goal_jpos": goal_joint_pos.tolist(),  "gmat": goal_mat.tolist(), "gquat": goal_quat.tolist(), "chosen_grasp_point": chosen_grasp_point.tolist(), "grasp_approach_delta": grasp_approach_delta.tolist()}
    # dict_file = {"chosen_grasp_i": chosen_grasp_i+1, "gpos": [float(goal_pos[0]), float(goal_pos[1]), float(goal_pos[2])], "pre_grasp": [float(pre_grasp_arr[chosen_grasp_i][0]), float(pre_grasp_arr[chosen_grasp_i][1]), float(pre_grasp_arr[chosen_grasp_i][2])], "approach": [float(approach_arr[chosen_grasp_i][0]), float(approach_arr[chosen_grasp_i][1]), float(approach_arr[chosen_grasp_i][2])],  "gmat": [[float(goal_mat[0,0]), float(goal_mat[0,1]), float(goal_mat[0,2])], [float(goal_mat[1,0]), float(goal_mat[1,1]), float(goal_mat[1,2])], [float(goal_mat[2,0]), float(goal_mat[2,1]), float(goal_mat[2,2])]], "gquat": [float(goal_quat[0]), float(goal_quat[1]), float(goal_quat[2]), float(goal_quat[3])]}
    with open(r'/home/harshit/meta/storm/polygrasp_goal.yml', 'w') as file:
        documents = yaml.dump(dict_file, file,default_flow_style=None)
    with open(r'polygrasp_goal.yml', 'w') as file:
        documents = yaml.dump(dict_file, file,default_flow_style=None)
    cameras.sub = None
    # grasp_client.visualize_grasp(scene_pcd, final_filtered_grasps, n=5)
    # grasp_client.visualize_grasp(obj_pcds[obj_i], final_filtered_grasps, name="obj", n=5)
    system("cd ~/meta/fairo/perception/sandbox/polygrasp/ && export CUDA_HOME=usr/local/cuda && mrp down grasp_server")
    # traj = execute_grasp(robot, chosen_grasp, hori_offset, time_to_go)
if __name__ == "__main__":
    main()
