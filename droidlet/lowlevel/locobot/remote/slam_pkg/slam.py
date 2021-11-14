# need to convert it to api
from pyrobot import Robot

import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from pyrobot.utils.util import try_cv2_import
import argparse
from scipy import ndimage
from copy import deepcopy as copy
import time
from math import ceil, floor, radians
import sys
import json
from pyrobot.locobot.base_control_utils import LocalActionStatus
import random
import shutil

cv2 = try_cv2_import()

# for slam modules
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from skimage.morphology import disk, binary_dilation
from slam_pkg.utils.map_builder import MapBuilder as mb
from slam_pkg.utils.fmm_planner import FMMPlanner
from slam_pkg.utils import depth_util as du

class TrackBack(object):
    def __init__(self):
        self.locs = set()

    def update(self, loc):
        self.locs.add(loc)

    def dist(self, a, b):
        d = np.linalg.norm((np.array(a) - np.array(b)), ord=1)
        # print(f'dist {a, b} = {d}')
        return d

    def get_loc(self, cur_loc, traversable):
        ans = None
        d = 10000000
        cand = [x for x in self.locs if traversable[round(x[1]), round(x[0])]]
        # for x in self.locs:
        #     if not traversable[round(x[1]), round(x[0])]:
        #         print(f'removing {x} not traversable')
        #         self.locs.remove(x)
        for x in cand:
            # print(f'candidate loc {round(x[0]), round(x[1])}, cur_loc {cur_loc}')
            if d > self.dist(cur_loc, x):
                ans = x
                d = self.dist(cur_loc, x)
        print(f'track back loc {ans}')
        return ans

class LabelPropSaver:
    def __init__(self, root):
        self.save_folder = root
        self.img_folder = os.path.join(self.save_folder, "rgb")
        self.img_folder_dbg = os.path.join(self.save_folder, "rgb_dbg")
        self.depth_folder = os.path.join(self.save_folder, "depth")
        self.seg_folder = os.path.join(self.save_folder, "seg")
        self.trav_folder = os.path.join(self.save_folder, "trav")

        if os.path.exists(self.save_folder):
            shutil.rmtree(self.save_folder)

        for x in [self.save_folder, self.img_folder, self.img_folder_dbg, self.depth_folder, self.seg_folder, self.trav_folder]:
            self.create(x)

        self.pose_dict = {}
        self.save_vis_skip_frames = 0
        self.img_count = 0
        self.dbg_str = "None"

    def create(self, d):
        if not os.path.isdir(d):
            os.makedirs(d)
    
    def get_total_frames(self):
        return self.img_count

    def save(self, rgb, depth, seg, pos):
        self.save_vis_skip_frames += 1
        if self.save_vis_skip_frames % 10 == 0:
            # store the images and depth
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            cv2.imwrite(
                self.img_folder + "/{:05d}.jpg".format(self.img_count),
                rgb,
            )

            cv2.putText(rgb, self.dbg_str, (40,40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))

            # robot_dbg_str = 'robot_pose ' + str(np.round(self.get_robot_global_state(), 3))
            # cv2.putText(rgb, robot_dbg_str, (40,60), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))

            cv2.imwrite(
                self.img_folder_dbg + "/{:05d}.jpg".format(self.img_count),
                rgb,
            )

            # store depth in mm
            depth *= 1e3
            depth[depth > np.power(2, 16) - 1] = np.power(2, 16) - 1
            depth = depth.astype(np.uint16)
            np.save(self.depth_folder + "/{:05d}.npy".format(self.img_count), depth)

            # store seg
            np.save(self.seg_folder + "/{:05d}.npy".format(self.img_count), seg)

            # store pos
            self.pose_dict[self.img_count] = copy(pos)
            self.img_count += 1
            
            # print(f"img_count {self.img_count}, #active {self.active_count}, self.goal_loc {self.goal_loc}, base_pos {pos}")
            with open(os.path.join(self.save_folder, "data.json"), "w") as fp:
                json.dump(self.pose_dict, fp)

class Slam(object):
    def __init__(
        self,
        robot,
        robot_name,
        map_size=4000,
        resolution=5,
        robot_rad=25,
        agent_min_z=5,
        agent_max_z=70,
        vis=False,
        save_vis=os.getenv("SAVE_VIS", 'False').lower() in ('true', 'True'),
        save_folder=os.getenv("SLAM_SAVE_FOLDER", '../slam_logs'),
    ):
        """

        :param robot: pyrobot robot object, only supports [habitat, locobot]
        :param robot_name: name of the robot [habitat, locobot] 
        :param map_size: size of map to be build in cm, assumes square map
        :param resolution: resolution of map, 1 pix = resolution distance(in cm) in real world
        :param robot_rad: radius of the agent, used to explode the map
        :param agent_min_z: robot min z (in cm), depth points below this will be considered as free space
        :param agent_max_z: robot max z (in cm), depth points above this will be considered as free space
        :param vis: whether to show visualization
        :param save_vis: whether to save visualization
        :param save_folder: path to save visualization

        :type robot: pytobot.Robot
        :type robot_name: str
        :type map_size: int
        :type resolution: int
        :type robot_rad: int
        :type agent_min_z: int
        :type agent_max_z: int
        :type vis: bool
        :type save_vis: bool
        :type save_folder: str
        """
        self.robot = robot
        self.robot_name = robot_name
        self.robot_rad = robot_rad
        self.map_builder = mb(
            map_size_cm=map_size,
            resolution=resolution,
            agent_min_z=agent_min_z,
            agent_max_z=agent_max_z,
        )
        self.maxx = 0
        self.maxy = 0
        self.minx = 0
        self.miny = 0
        # initialize variable
        robot.camera.reset()
        time.sleep(2)

        self.init_state = self.get_robot_global_state()
        self.prev_bot_state = (0, 0, 0)
        self.col_map = np.zeros((self.map_builder.map.shape[0], self.map_builder.map.shape[1]))
        self.robot_loc_list_map = np.array(
            [
                self.real2map(
                    self.get_rel_state(self.get_robot_global_state(), self.init_state)[:2]
                )
            ]
        )
        self.map_builder.update_map(
            self.robot.camera.get_current_pcd(in_cam=False)[0],
            self.get_rel_state(self.get_robot_global_state(), self.init_state),
        )

        # for visualization purpose #
        self.init_save(save_folder, save_vis)
        self.root_folder = save_folder
        self.vis = vis
        # to visualize robot heading
        triangle_scale = 0.5
        self.triangle_vertex = np.array([[0.0, 0.0], [-2.0, 1.0], [-2.0, -1.0]])
        self.triangle_vertex *= triangle_scale
        
        self.last_pos = self.robot.base.get_state()

        # for bumper check of locobot
        if self.robot_name == "locobot":
            print(f'robot_name {self.robot_name}')
            from slam_pkg.utils.locobot_bumper_checker import BumperCallbacks

            self.bumper_state = BumperCallbacks()
            print(f'self.bumper_state {self.bumper_state}')
            # for mapping refer to http://docs.ros.org/groovy/api/kobuki_msgs/html/msg/BumperEvent.html
            self.bumper_num2ang = {0: np.deg2rad(30), 1: 0, 2: np.deg2rad(-30)}

        self.whole_area_explored = True
        self.last_stg = None
        self.explore_goal = None
        self.debug_state = {}
        self.track_back = TrackBack()

    def init_save(self, save_folder, save_vis=True):
        self.save_vis = save_vis
        self.start_vis = False
        self.save_folder = save_folder
        print(f"save_vis {save_vis}, save_folder {save_folder}")
        self.last_base_pos = None
        self.default_saver = LabelPropSaver(os.path.join(save_folder, 'default'))
        self.active_saver = LabelPropSaver(os.path.join(save_folder, 'activeonly'))
        
        self.vis_count = 0
        
        self.exec_wait = not (self.save_vis)
        print(f'exec_wait {self.exec_wait}')

        self.maxx = 0
        self.maxy = 0
        self.minx = 0
        self.miny = 0
        self.objects_explored = 0

    def set_dbg_str(self, x):
        print(f'dbg str {x}')
        self.active_saver.dbg_str = x
        self.default_saver.dbg_str = x

    def set_explore_goal(self, goal):
        print(f'setting explore goal {goal}')
        self.explore_goal = goal

    def set_goal(self, goal):
        """
        goal is 3 len tuple with position in real world in robot start frame
        :param goal: goal to be reached in metric unit

        :type goal: tuple

        :return:
        """
        self.goal_loc = goal
        self.goal_loc_map = self.real2map(self.goal_loc[:2])
        print(f'set_goal {self.goal_loc, self.goal_loc_map, goal}')

    def set_relative_goal_in_robot_frame(self, goal):
        """
        goal is 3 len tuple with position in real world in robot current frmae
        :param goal: goal to be reached in metric unit

        :type goal: tuple

        :return:
        """
        robot_pr_pose = self.get_robot_global_state()
        # check this part
        abs_pr_goal = list(self.get_rel_state(goal, (0.0, 0.0, -robot_pr_pose[2])))
        abs_pr_goal[0] += robot_pr_pose[0]
        abs_pr_goal[1] += robot_pr_pose[1]
        abs_pr_goal[2] = goal[2] + robot_pr_pose[2]

        # convert the goal in init frame
        self.goal_loc = self.get_rel_state(abs_pr_goal, self.init_state)
        self.goal_loc_map = self.real2map(self.goal_loc[:2])

        # TODO: make it non blocking
        while self.take_step(25) is None:
            print(f'set_relative_goal_in_robot_frame')
            continue

    def set_absolute_goal_in_robot_frame(self, goal):
        """
        goal is 3 len tuple with position in real world in robot start frmae
        :param goal: goal to be reached in metric unit

        :type goal: tuple

        :return:
        """
        # convert the relative goal to abs goal
        if abs(goal[0]) >= 20 or abs(goal[1]) >= 20:
            print(f'set_absolute_goal_in_robot_frame skipping out of bounds goal {goal}')
            return
        self.objects_explored += 1
        self.goal_loc = self.get_rel_state(goal, self.init_state)
        # convert the goal in inti frame
        self.goal_loc_map = self.real2map(self.goal_loc[:2])
        print(f'set_absolute_goal_in_robot_frame {self.goal_loc, self.goal_loc_map, goal}')
        # TODO make it non blocking
        while self.take_step(25) is None:
            print(f'set_absolute_goal_in_robot_frame')
            continue

    def update_map(self):
        """Updtes map , explode it by the radius of robot, add collison map to it and return the traversible area

        Returns:
            [np.ndarray]: [traversible space]
        """
        robot_state = self.get_rel_state(self.get_robot_global_state(), self.init_state)
        self.map_builder.update_map(
            self.robot.camera.get_current_pcd(in_cam=False)[0], robot_state
        )

        # explore the map by robot shape
        obstacle = self.map_builder.map[:, :, 1] >= 1.0
        selem = disk(self.robot_rad / self.map_builder.resolution)
        traversable = binary_dilation(obstacle, selem) != True

        return traversable
    
    def get_stg(self, step_size):
        traversable = self.update_map()
        self.planner = FMMPlanner(
            traversable, step_size=int(step_size / self.map_builder.resolution)
        )
        self.planner.set_goal(self.goal_loc_map)
        robot_map_loc = self.real2map(
            self.get_rel_state(self.get_robot_global_state(), self.init_state)
        )
        self.stg = self.planner.get_short_term_goal(robot_map_loc)
        return traversable

    def take_step(self, step_size):
        """
        step size in meter
        :param step_size:
        :return:
        """
        print(f'\nstep begin ...')
        traversable = self.get_stg(step_size) # sets self.stg

        # print(f'self.goal_loc {self.goal_loc} self.explore_goal {self.explore_goal}')
        # convert goal from map space to robot space
        stg_real = self.map2real(self.stg)

        def pp(n, t):
            print(f'{n}, {round(t[0]), round(t[1])} ')

        pp('goal_loc', self.goal_loc)
        pp('goal_loc_map', self.goal_loc_map)
        pp('stg', self.stg)

        # convert stg real from init frame to global frame of pyrobot
        stg_real_g = self.get_absolute_goal((stg_real[0], stg_real[1], 0))
        robot_state = self.get_rel_state(self.get_robot_global_state(), self.init_state)
        robot_map_loc = self.real2map(robot_state)

        pp(f'robot_map_loc before translation', robot_map_loc)
        print(f'self.planner.fmm_dist[{self.stg[1]}][{self.stg[0]}] = {self.planner.fmm_dist[self.stg[1], self.stg[0]]}')
        # print(f'self.planner.fmm_dist[{self.stg[0]}][{self.stg[1]}] = {self.planner.fmm_dist[self.stg[0], self.stg[1]]}')

        # check whether goal is on collision # should never happen? 
        if not traversable[self.stg[1], self.stg[0]]:
            print("Obstacle in path! Should never happen, stg should never be an obstacle!!")
            print(f'traversable[stg] {traversable[self.stg[1], self.stg[0]]}')
            # print(f'robot_map_loc {robot_map_loc} traversable.shape {traversable.shape}')
            print(f'traversable[robot_loc] {traversable[round(robot_map_loc[1]), round(robot_map_loc[0])]}')
            print("Goal Not reachable")
            if self.goal_loc == self.explore_goal: # stuck in explore
                self.whole_area_explored = True
                self.save_end_state("stg is not traversable", traversable)
            return False
        else:            
            # go to the location the robot
            y = np.arctan2(stg_real[1] - robot_state[1], stg_real[0] - robot_state[0])
            # print(f'stg_real[1] - robot_state[1] {stg_real[1] - robot_state[1]}, stg_real[0] - robot_state[0] {stg_real[0] - robot_state[0]}')
            print(f'yaw for move {y}, robot_state {robot_state}, stg_real {stg_real}')
            exec = self.robot.base.go_to_absolute(
                (
                    stg_real_g[0],
                    stg_real_g[1],
                    y,
                ),
                wait=self.exec_wait,
            )
            while self.robot.base._as.get_state() == LocalActionStatus.ACTIVE:
                if self.save_vis:
                    self.save_rgb_depth_seg()
                else:
                    pass
        
        exec = self.robot.base._as.get_state() == LocalActionStatus.SUCCEEDED
        robot_map_loc = self.real2map(
            self.get_rel_state(self.get_robot_global_state(), self.init_state)
        )
        if exec:
            # if self.robot.base._as.get_state() == LocalActionStatus.SUCCEEDED:
            print(f'finished translation')
            pp('robot_map_loc', robot_map_loc)
            self.track_back.update(robot_map_loc)
        else:
            print(f'translation failed') 
            pp('robot_map_loc', robot_map_loc)

            # print(f'robot_map_loc {robot_map_loc} traversable.shape {traversable.shape}')
            print(f'is robot_loc traversable {traversable[round(robot_map_loc[1]), round(robot_map_loc[0])]}')
            print(f'is stg traversable {traversable[self.stg[1], self.stg[0]]}')

            # set map builder as obstacle 
            # TODO use robot_map_loc to update map builder instead of stg (which might be faraway and traversable)
            # self.map_builder.map[round(robot_map_loc[1]), round(robot_map_loc[0]), 1] = 1
            self.map_builder.map[self.stg[1], self.stg[0], 1] = 1
            ostg = self.stg
            # print(f'map_builder loc update {self.map_builder.map[round(robot_map_loc[1]), round(robot_map_loc[0]), 1]}')
            
            # traversable = self.update_map()
            traversable = self.get_stg(step_size)
            print(f'ostg {ostg}, stg {self.stg}')
            #check here that fmm_dist is updated 

            ob = [x for x in zip(*np.where(self.planner.fmm_dist == 10000))]
            if (ostg[1], ostg[0]) in ob:
                print(f'stg added to {len(ob)} obstaceles')
            else:
                print(f'stg not found in {len(ob)} obstacles')

            print(f'is robot_loc traversable after update {traversable[round(robot_map_loc[1]), round(robot_map_loc[0])]}')
            print(f'is stg traversable after update {traversable[ostg[1], ostg[0]]}')
            print(f'self.planner.fmm_dist[{ostg[1]}][{ostg[0]}] = {self.planner.fmm_dist[ostg[1], ostg[0]]}')
            # track back 
            track_back = self.map2real(self.track_back.get_loc(robot_map_loc, traversable))
            track_back_g = self.get_absolute_goal((track_back[0], track_back[1], 0))
            self.robot.base.go_to_absolute(track_back_g, wait=self.exec_wait)
            while self.robot.base._as.get_state() == LocalActionStatus.ACTIVE:
                if self.save_vis:
                    self.save_rgb_depth_seg()
                else:
                    pass
            if self.robot.base._as.get_state() == LocalActionStatus.SUCCEEDED:
                print(f'track back succeeded to {self.track_back}')
                robot_map_loc = self.real2map(
                    self.get_rel_state(self.get_robot_global_state(), self.init_state)
                )
                pp('robot_map_loc', robot_map_loc)
                # print(f'robot_map_loc {round(robot_map_loc[0]), round(robot_map_loc[1])}')
            else:
                print('track back failed') # possible mode of failure. shouldn't happen? check in noisy setting.

        robot_state = self.get_rel_state(self.get_robot_global_state(), self.init_state)
        # print("bot_state after executing action = {}".format(robot_state))

        # update robot location list
        robot_state_map = self.real2map(robot_state[:2])
        self.robot_loc_list_map = np.concatenate(
            (self.robot_loc_list_map, np.array([robot_state_map]))
        )
        self.prev_bot_state = robot_state
        
        self.visualize()

        def get_rounded_map_dist(a, b):
            a = np.array([round(x) for x in a])
            b = np.array([round(x) for x in b])
            return np.linalg.norm(a-b)

        # print(f'robot_state_map {robot_state_map, self.goal_loc_map, np.array(robot_state_map)}')
        print(f'distance to goal {get_rounded_map_dist(robot_state_map, self.goal_loc_map)}')
        # return True if robot reaches within threshold
        if (get_rounded_map_dist(robot_state_map, self.goal_loc_map) == 0):
            print("robot has reached goal")
            return True

        # return False if goal is not reachable
        if not traversable[round(self.goal_loc_map[1]), round(self.goal_loc_map[0])]:
            print("Goal Not reachable")
            return False
        if (
            self.planner.fmm_dist[round(robot_state_map[1]), round(robot_state_map[0])]
            >= self.planner.fmm_dist.max()
        ):
            print("whole area is explored")
            self.whole_area_explored = True
            self.save_end_state("fmmdist >= max", traversable)
            return False
        return None
    
    def save_end_state(self, msg, traversable):
        np.save(self.default_saver.trav_folder + "/traversable.npy", traversable)
        robot_state = self.get_rel_state(self.get_robot_global_state(), self.init_state)
        robot_state_map = self.real2map(robot_state[:2])
        self.debug_state = {
                "msg": msg,
                "stg": [round(self.stg[0]), round(self.stg[1])],
                "robot_state_map": [round(robot_state_map[0]), round(robot_state_map[1])],
                "planner.fmm_dist": float(self.planner.fmm_dist[round(robot_state_map[1]), round(robot_state_map[0])]),
                "planner.fmm_dist.max": float(self.planner.fmm_dist.max()),
                "save_folder": self.save_folder,
                "area_explored": float(self.get_area_explored()[0]),
                "objects_explored": self.objects_explored,
                "active_frames": self.active_saver.get_total_frames(),
                "default_frames": self.default_saver.get_total_frames(),

            }
        for k, v in self.debug_state.items():
            print(f'{v, type(v)}')
        print(f'debug_state {self.debug_state}')

    def get_area_explored(self):
        return abs(self.maxx - self.minx) * abs(self.maxy - self.miny)

    # def save_all(self):
    #     self.skp += 1
    #     # print(f'goal_loc {self.goal_loc} explore_goal {self.explore_goal}')
    #     if pos != self.last_pos and self.skp % 10 == 0:
    #         self.last_pos = pos
    #         # store the images and depth
    #         rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    #         cv2.imwrite(
    #             self.img_folder + "/{:05d}.jpg".format(self.img_count),
    #             rgb,
    #         )

    #         cv2.putText(rgb, self.dbg_str, (40,40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))

    #         # robot_dbg_str = 'robot_pose ' + str(np.round(self.get_robot_global_state(), 3))
    #         # cv2.putText(rgb, robot_dbg_str, (40,60), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))

    #         cv2.imwrite(
    #             self.img_folder_dbg + "/{:05d}.jpg".format(self.img_count),
    #             rgb,
    #         )

    #         # store depth in mm
    #         depth *= 1e3
    #         depth[depth > np.power(2, 16) - 1] = np.power(2, 16) - 1
    #         depth = depth.astype(np.uint16)
    #         np.save(self.depth_folder + "/{:05d}.npy".format(self.img_count), depth)

    #         # store seg
    #         np.save(self.seg_folder + "/{:05d}.npy".format(self.img_count), seg)

    #         # store pos
    #         self.pos_dic[self.img_count] = copy(pos)
    #         self.img_count += 1
            
    #         self.active_count += is_active
    #         # print(f"img_count {self.img_count}, #active {self.active_count}, self.goal_loc {self.goal_loc}, base_pos {pos}")
    #         with open(os.path.join(self.save_folder, "data.json"), "w") as fp:
    #             json.dump(self.pos_dic, fp)


    def save_rgb_depth_seg(self):
        rgb, depth, seg = self.robot.camera.get_rgb_depth_segm()
        pos = self.robot.base.get_state()
        if pos != self.last_base_pos:
            self.last_base_pos = pos
            is_active = 0 if self.goal_loc == self.explore_goal else 1
            if is_active:
                self.active_saver.save(rgb, depth, seg, pos)
            self.default_saver.save(rgb, depth, seg, pos)

    def get_absolute_goal(self, loc):
        """
        Transfer loc in init robot frame to global frame
        :param loc: location in init frame in metric unit

        :type loc: tuple

        :return: location in global frame in metric unit
        :rtype: list
        """
        # 1) orient goal to global frame
        loc = self.get_rel_state(loc, (0.0, 0.0, -self.init_state[2]))

        # 2) add the offset
        loc = list(loc)
        loc[0] += self.init_state[0]
        loc[1] += self.init_state[1]
        return tuple(loc)

    def real2map(self, loc):
        """
        convert real world location to map location
        :param loc: real world location in metric unit

        :type loc: tuple

        :return: location in map space
        :rtype: tuple [x_map_pix, y_map_pix]
        """
        # converts real location to map location
        loc = np.array([loc[0], loc[1], 0])
        loc *= 100  # convert location to cm
        map_loc = du.transform_pose(
            loc,
            (self.map_builder.map_size_cm / 2.0, self.map_builder.map_size_cm / 2.0, np.pi / 2.0),
        )
        map_loc /= self.map_builder.resolution
        map_loc = map_loc.reshape(3)
        return tuple(map_loc[:2])

    def map2real(self, loc):
        """
        convert map location to real world location
        :param loc: map location [x_pixel_location, y_pixel_location]

        :type loc: list

        :return: corresponding map location in real world in metric unit
        :rtype: list [x_real_world, y_real_world]
        """
        # converts map location to real location
        loc = np.array([loc[0], loc[1], 0])
        real_loc = du.transform_pose(
            loc,
            (
                -self.map_builder.map.shape[0] / 2.0,
                self.map_builder.map.shape[1] / 2.0,
                -np.pi / 2.0,
            ),
        )
        real_loc *= self.map_builder.resolution  # to take into account map resolution
        real_loc /= 100  # to convert from cm to meter
        real_loc = real_loc.reshape(3)
        return real_loc[:2]

    def get_rel_state(self, cur_state, init_state):
        """
        helpful for calculating the relative state of cur_state wrt to init_state [both states are wrt same frame]
        :param cur_state: frame for which position to be calculated
        :param init_state: frame in which position to be calculated

        :type cur_state: tuple [x_robot, y_robot, yaw_robot]
        :type init_state: tuple [x_robot, y_robot, yaw_robot]

        :return: relative state of cur_state wrt to init_state
        :rtype list [x_robot_rel, y_robot_rel, yaw_robot_rel]
        """
        # get relative in global frame
        rel_X = cur_state[0] - init_state[0]
        rel_Y = cur_state[1] - init_state[1]
        # transfer from global frame to init frame
        R = np.array(
            [
                [np.cos(init_state[2]), np.sin(init_state[2])],
                [-np.sin(init_state[2]), np.cos(init_state[2])],
            ]
        )
        rel_x, rel_y = np.matmul(R, np.array([rel_X, rel_Y]).reshape(-1, 1))
        self.maxx = max(self.maxx, rel_x)
        self.maxy = max(self.maxy, rel_y)
        self.minx = min(self.minx, rel_x)
        self.miny = min(self.miny, rel_y)

        return rel_x[0], rel_y[0], cur_state[2] - init_state[2]

    def get_robot_global_state(self):
        """
        :return: return the global state of the robot [x_robot_loc, y_robot_loc, yaw_robot]
        :rtype: tuple
        """
        return self.robot.base.get_state("odom")

    def visualize(self):
        """

        :return:
        """

        def vis_env_agent_state():
            # goal
            plt.plot(self.goal_loc_map[0], self.goal_loc_map[1], "y*")
            # short term goal
            plt.plot(self.stg[1], self.stg[0], "b*")
            # plt.plot(self.stg[0], self.stg[1], "b*")
            plt.plot(self.robot_loc_list_map[:, 0], self.robot_loc_list_map[:, 1], "r--")

            # draw heading of robot
            robot_state = self.get_rel_state(self.get_robot_global_state(), self.init_state)
            R = np.array(
                [
                    [np.cos(robot_state[2]), np.sin(robot_state[2])],
                    [-np.sin(robot_state[2]), np.cos(robot_state[2])],
                ]
            )
            global_tri_vertex = np.matmul(R.T, self.triangle_vertex.T).T
            map_global_tra_vertex = np.array(
                [
                    self.real2map((x[0] + robot_state[0], x[1] + robot_state[1]))
                    for x in global_tri_vertex
                ]
            )
            t1 = plt.Polygon(map_global_tra_vertex, color="red")
            plt.gca().add_patch(t1)

        if not self.start_vis:
            plt.figure(figsize=(40, 8))
            self.start_vis = True
        plt.clf()
        num_plots = 4

        # visualize RGB image
        plt.subplot(1, num_plots, 1)
        plt.title("RGB")
        plt.imshow(self.robot.camera.get_rgb())

        # visualize Depth image
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, num_plots, 2)
        plt.title("Depth")
        plt.imshow(self.robot.camera.get_depth())

        # visualize distance to goal & map, robot current location, goal, short term goal, robot path #
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, num_plots, 3)
        plt.title("Dist to Goal")
        plt.imshow(self.planner.fmm_dist, origin="lower")
        vis_env_agent_state()

        plt.subplot(1, num_plots, 4)
        plt.title("Map")
        plt.imshow(self.map_builder.map[:, :, 1] >= 1.0, origin="lower")
        vis_env_agent_state()

        plt.gca().set_aspect("equal", adjustable="box")
        if self.save_vis:
            plt.savefig(os.path.join(self.save_folder, "{:04d}.jpg".format(self.vis_count)))
        if self.vis:
            plt.pause(0.1)
        self.vis_count += 1


def main(args):
    if args.robot == "habitat":
        assets_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../test/test_assets")
        )
        scene = args.scene
        time.sleep(10.0)
        try:
            config = {
                "physics_config": os.path.join(assets_path, "default.phys_scene_config.json"),
                "scene_path": "{}/{}/habitat/mesh_semantic.ply".format(args.dataset_path, scene),
            }
            args.store_path = "./tmp/{}".format(scene)
            robot = Robot("habitat", common_config=config)
            agent_state = robot.base.agent.get_state()
            # place the robot at place where it can move
            p = robot.base.sim.pathfinder.get_random_navigable_point()

            agent_state.position = copy(p)
            agent_state.sensor_states["rgb"].position = copy(p + np.array([0.0, 0.6, 0.0]))
            agent_state.sensor_states["depth"].position = copy(p + np.array([0.0, 0.6, 0.0]))
            agent_state.sensor_states["semantic"].position = copy(p + np.array([0.0, 0.6, 0.0]))
            robot.base.agent.set_state(agent_state)
            print("trying scene = {}".format(scene))

            slam = Slam(
                robot,
                args.robot,
                args.map_size,
                args.resolution,
                args.robot_rad,
                args.agent_min_z,
                args.agent_max_z,
                args.vis,
                args.save_vis,
                args.store_path,
            )
            slam.set_goal(tuple(args.goal))
            while slam.take_step(step_size=args.step_size) is None:
                slam.visualize()

            # save pos dic
            with open(os.path.join(slam.save_folder, "data.json"), "w") as fp:
                json.dump(slam.pos_dic, fp)
            slam.visualize()

            """
            # helpful visualizing the exploration
            #TODO: to make it work need to install ffmpeg to cker image
            # generate gif out of plt images
            os.system(
                "ffmpeg -framerate 6 -f image2 -i {}/%04d.jpg {}/exploration.gif".format(
                    slam.save_folder, slam.save_folder
                )
            )

            # rm plt images
            os.system("rm {}/*.jpg".format(slam.save_folder))
            """
        except:
            print("not able to open the scene = {}".format(scene))

    elif args.robot == "locobot":
        robot = Robot("locobot")
        slam = Slam(
            robot,
            args.robot,
            args.map_size,
            args.resolution,
            args.robot_rad,
            args.agent_min_z,
            args.agent_max_z,
            args.vis,
            args.save_vis,
            args.store_path,
        )
        slam.set_goal(tuple(args.goal))
        while slam.take_step(step_size=args.step_size) is None:
            slam.visualize()

        # save pos dic
        with open(os.path.join(slam.save_folder, "data.json"), "w") as fp:
            json.dump(slam.pos_dic, fp)
        slam.visualize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for testing simple SLAM algorithm")
    parser.add_argument(
        "--robot", help="Name of the robot [locobot, habitat]", type=str, default="habitat"
    )
    parser.add_argument(
        "--goal", help="goal the robot should reach in metric unit", nargs="+", type=float
    )
    parser.add_argument("--map_size", help="lenght and with of map in cm", type=int, default=4000)
    parser.add_argument(
        "--resolution", help="per pixel resolution of map in cm", type=int, default=5
    )
    parser.add_argument("--step_size", help="step size in cm", type=int, default=25)
    parser.add_argument("--robot_rad", help="robot radius in cm", type=int, default=25)
    parser.add_argument("--agent_min_z", help="agent min height in cm", type=int, default=5)
    parser.add_argument("--agent_max_z", help="robot max height in cm", type=int, default=70)
    parser.add_argument("--vis", help="whether to show visualization", action="store_true")
    parser.add_argument("--save_vis", help="whether to store visualization", action="store_true")
    parser.add_argument(
        "--store_path", help="path to store visualization", type=str, default="./tmp"
    )
    parser.add_argument(
        "--dataset_path",
        help="path where Replica dataset is stored",
        type=str,
        default="/Replica-Dataset",
    )
    parser.add_argument(
        "--scene", help="scence name for data collection", type=str, default="apartment_0"
    )
    args = parser.parse_args()
    main(args)