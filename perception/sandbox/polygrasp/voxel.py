import logging
import json
from time import time
from time import sleep
from types import SimpleNamespace

import numpy as np
import open3d as o3d

import a0
from polygrasp import serdes

from transforms3d.euler import euler2mat
import yaml
import xml.etree.cElementTree as ET

log = logging.getLogger(__name__)
topic = "cams/rgbd"
# topic = "color_image_compressed"


class CameraSubscriber:
    def __init__(self, intrinsics_file, extrinsics_file):
        with open(intrinsics_file, "r") as f:
            intrinsics_json = json.load(f)
            self.intrinsics = [SimpleNamespace(**d) for d in intrinsics_json]

        with open(extrinsics_file, "r") as f:
            self.extrinsics = json.load(f)
        print(len(self.intrinsics))
        print(len(self.extrinsics))
        assert len(self.intrinsics) == len(self.extrinsics)
        self.n_cams = len(self.intrinsics)
        self.done = False
        self.rgbds = None
        self.sub = None
        # self.sub = a0.RemoteSubscriber("172.16.0.3", topic, self.cback, a0.INIT_AWAIT_NEW, a0.ITER_NEXT)
        self.sub = a0.RemoteSubscriber("172.16.0.3", topic, self.cback, a0.INIT_AWAIT_NEW, a0.ITER_NEWEST)
        # self.sub = a0.RemoteSubscriber("172.16.0.3", topic, self.cback, a0.INIT_MOST_RECENT, a0.ITER_NEWEST)
        # self.sub = a0.SubscriberSync(topic, a0.INIT_MOST_RECENT, a0.ITER_NEWEST)
        self.recent_rgbd = None

    def cback(self, pkt):
        # self.done = True
        self.recent_rgbd = serdes.bytes_to_np(pkt.payload)
        if ((self.recent_rgbd is not None) and len(self.recent_rgbd)>0):
            self.done = True
        # print(f"Got {pkt.payload}")
        # print("Got message")

    def get_rgbd(self):
        if self.sub.can_read():
            self.recent_rgbd = serdes.bytes_to_np(self.sub.read().payload)
        return self.recent_rgbd


class PointCloudSubscriber(CameraSubscriber):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.width = self.intrinsics[0].width
        self.height = self.intrinsics[0].height

        # Convert to open3d intrinsics
        self.o3_intrinsics = [
            o3d.camera.PinholeCameraIntrinsic(
                width=intrinsic.width,
                height=intrinsic.height,
                fx=intrinsic.fx,
                fy=intrinsic.fy,
                cx=intrinsic.ppx,
                cy=intrinsic.ppy,
            )
            for intrinsic in self.intrinsics
        ]

        # Convert to numpy homogeneous transforms
        self.extrinsic_transforms = np.empty([self.n_cams, 4, 4])
        for i, calibration in enumerate(self.extrinsics):
            self.extrinsic_transforms[i] = np.eye(4)
            self.extrinsic_transforms[i, :3, :3] = calibration["camera_base_ori"]
            self.extrinsic_transforms[i, :3, 3] = calibration["camera_base_pos"]

    def get_pcd_i(self, rgbd: np.ndarray, cam_i: int, mask: np.ndarray = None):
        if mask is None:
            mask = np.ones([self.height, self.width])

        intrinsic = self.o3_intrinsics[cam_i]
        transform = self.extrinsic_transforms[cam_i]

        # The specific casting here seems to be very important, even though
        # rgbd should already be in np.uint16 type...
        img = (rgbd[:, :, :3] * mask[:, :, None]).astype(np.uint8)
        depth = (rgbd[:, :, 3] * mask).astype(np.uint16)

        o3d_img = o3d.cuda.pybind.geometry.Image(img)
        o3d_depth = o3d.cuda.pybind.geometry.Image(depth)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_img,
            o3d_depth,
            convert_rgb_to_intensity=False,
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        pcd.transform(transform)

        return pcd

    def get_pcd(
        self, rgbds: np.ndarray, masks: np.ndarray = None
    ) -> o3d.geometry.PointCloud:
        if masks is None:
            masks = np.ones([self.n_cams, self.height, self.width])
        pcds = [self.get_pcd_i(rgbds[i], i, masks[i]) for i in range(len(rgbds))]
        result = pcds[0]
        for pcd in pcds[1:]:
            result += pcd
        return result


if __name__ == "__main__":   
    pcs = PointCloudSubscriber("conf/intrinsics.json", "conf/extrinsics.json")
    # rgbds = pcs.get_rgbd()
    while(pcs.done != True):
        sleep(0.1)
        print("Waiting")
    print("Done waiting")
    rgbds = pcs.recent_rgbd
    merged_pc = pcs.get_pcd(rgbds)
    # o3d.io.write_point_cloud("./m.ply", merged_pc)
    o3d.io.write_point_cloud("../../../../storm/m.ply", merged_pc)
    print("Saving to m.ply...")
    pcs.sub = None

    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud("../../../../storm/m.ply")
    print(pcd)
    # o3d.visualization.draw_geometries([pcd])
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(0.15, -0.75, 0.0), max_bound=(0.9, 0.750, 0.95))
    bounding_box.color = (1, 0, 0)
    # Draw the newly cropped PCD and bounding box
    # o3d.visualization.draw_geometries([pcd, bounding_box])
    cropped_pcd = pcd.crop(bounding_box)
    # o3d.visualization.draw_geometries([cropped_pcd])
    orig_load = np.asarray(cropped_pcd.points)
    voxel_size = 0.03 #0.1
    print('voxelization')
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cropped_pcd, voxel_size=voxel_size)
    # o3d.visualization.draw_geometries([voxel_grid])
    tree = ET.parse('../../../../storm/template.xml')
    root = tree.getroot()
    pos_data = root[-1][2].attrib["pos"]
    CARTESIAN_OFFSET = [float(x) for x in pos_data.split()]
    CARTESIAN_OFFSET = np.array((CARTESIAN_OFFSET))
    euler_data = root[-1][2].attrib["euler"]
    robo_euler = [float(x) for x in euler_data.split()]
    # robo_euler = np.array((robo_euler))
    rot_mat = euler2mat(robo_euler[0], robo_euler[1], robo_euler[2])
    # Read YAML file
    with open("../../../../storm/template.yml", 'r') as stream:
        dict_file = yaml.safe_load(stream)
    voxels=voxel_grid.get_voxels()
    i=1
    num_default_coll_objs = 7
    num_horizon = 60
    for j in range(num_horizon):
        cube = ET.SubElement(root[-1], "site", pos='0 0 .10', euler='0 0 0', size='0.01', rgba='.8 .1 .2 .4', name="end_effector" + str(j))

    for v in voxels:
        center = (voxel_grid.get_voxel_center_coordinate(v.grid_index)).tolist()
        dict_file["world_model"]["coll_objs"]["cube"]["cube" + str(i+num_default_coll_objs)] = {"dims":[voxel_size, voxel_size, voxel_size], "pose": [center[0], center[1], center[2], 0.0, 0.0, 0.0, 1.0]}
        if(i<900):
            center = rot_mat@center + CARTESIAN_OFFSET
            cube = ET.SubElement(root[-1], "body", pos=str(center[0]) + " " + str(center[1]) + " " + str(center[2]), euler=euler_data, name="cube" + str(i))
            ET.SubElement(cube, "geom", type="box", size=str(voxel_size/2) + " " + str(voxel_size/2) + " " + str(voxel_size/2), pos="0.0 0.0 0.0", rgba=str(v.color[0])+" "+ str(v.color[1])+" "+ str(v.color[2])+" "+ "1.0", contype="0", conaffinity="0")
        i += 1
    with open(r'../../../../storm/content/configs/gym/collision_primitives_3d_pc.yml', 'w') as file:
        documents = yaml.dump(dict_file, file,default_flow_style=None)
    i -= 1
    print(i)
    tree.write("../../../../mj_envs/mj_envs/envs/arms/franka/assets/franka_reach_v0_pc.xml")