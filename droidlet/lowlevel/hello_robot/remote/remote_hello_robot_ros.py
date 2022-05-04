"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
# python -m Pyro4.naming -n <MYIP>
import select
import logging
import os
import json
import time
import copy
from math import *
from rich import print
import Pyro4
import numpy as np
from droidlet.lowlevel.hello_robot.remote.utils import transform_global_to_base, goto
from stretch_ros_move_api import MoveNode as Robot
from droidlet.lowlevel.pyro_utils import safe_call


Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.ITER_STREAMING = True


# #####################################################
@Pyro4.expose
class RemoteHelloRobot(object):
    """Hello Robot interface"""

    def __init__(self, ip):
        self._ip = ip
        self._robot = Robot()
        self._robot.start()
        self._done = True
        self.cam = None
        # Read battery maintenance guide https://docs.hello-robot.com/battery_maintenance_guide/
        self._load_urdf()
        self.tilt_correction = 0.0

    def _load_urdf(self):
        import os

        urdf_path = os.path.join(
            os.getenv("HELLO_FLEET_PATH"),
            os.getenv("HELLO_FLEET_ID"),
            "exported_urdf",
            "stretch.urdf",
        )

        from pytransform3d.urdf import UrdfTransformManager
        import pytransform3d.transformations as pt
        import pytransform3d.visualizer as pv
        import numpy as np

        self.tm = UrdfTransformManager()
        with open(urdf_path, "r") as f:
            urdf = f.read()
            self.tm.load_urdf(urdf)

    def set_tilt_correction(self, angle):
        """
        angle in radians
        """
        print(
            "[hello-robot] Setting tilt correction " "to angle: {} degrees".format(degrees(angle))
        )

        self.tilt_correction = angle

    def get_camera_transform(self):
        head_pan = self.get_pan()
        head_tilt = self.get_tilt()

        if self.tilt_correction != 0.0:
            head_tilt += self.tilt_correction

        # Get Camera transform
        self.tm.set_joint("joint_head_pan", head_pan)
        self.tm.set_joint("joint_head_tilt", head_tilt)
        camera_transform = self.tm.get_transform("camera_color_frame", "base_link")

        # correct for base_link's z offset from the ground
        # at 0, the correction is -0.091491526943
        # at 90, the correction is +0.11526719 + -0.091491526943
        # linear interpolate the correction of 0.023775
        interp_correction = 0.11526719 * abs(head_tilt) / radians(90)
        # print('interp_correction', interp_correction)

        camera_transform[2, 3] += -0.091491526943 + interp_correction

        return camera_transform

    def get_base_state(self):
        slam_pose = self._robot.get_slam_pose()
        x = slam_pose.pose.position.x
        y = slam_pose.pose.position.y
        theta = slam_pose.pose.orientation.z
        return (x, y, theta)

    def pull_status(self):
        pass

    def is_base_moving(self):
        time.sleep(10)
        return False

    def get_pan(self):
        return self._robot.get_joint_state("head_pan")

    def get_tilt(self):
        return self._robot.get_joint_state("head_tilt")

    def set_pan(self, pan):
        self._robot.send_command("joint_head_pan", pan)

    def set_tilt(self, tilt):
        self._robot.send_command("joint_head_tilt", tilt)

    def reset_camera(self):
        self.set_pan(0)
        self.set_tilt(0)

    def set_pan_tilt(self, pan, tilt):
        """Sets both the pan and tilt joint angles of the robot camera  to the
        specified values.

        :param pan: value to be set for pan joint in radian
        :param tilt: value to be set for the tilt joint in radian

        :type pan: float
        :type tilt: float
        :type wait: bool
        """
        self.set_pan(pan)
        self.set_tilt(tilt)

    def test_connection(self):
        print("Connected!!")  # should print on server terminal
        return "Connected!"  # should print on client terminal

    def push_command(self):
        pass

    def translate_by(self, x_m, v_m=None):
        self._robot.send_command("translate_mobile_base", x_m)

    def rotate_by(self, x_r):
        self._robot.send_command("rotate_mobile_base", x_r)

    def initialize_cam(self):
        if self.cam is None:
            # wait for realsense service to be up and running
            time.sleep(2)
            with Pyro4.Daemon(self._ip) as daemon:
                cam = Pyro4.Proxy("PYRONAME:hello_realsense@" + self._ip)
            self.cam = cam

    def go_to_absolute(self, xyt_position):
        """Moves the robot base to given goal state in the world frame.

        :param xyt_position: The goal state of the form (x,y,yaw)
                             in the world (map) frame.
        """
        status = "SUCCEEDED"
        if self._done:
            self.initialize_cam()
            self._done = False
            global_xyt = xyt_position
            base_state = self.get_base_state()
            base_xyt = transform_global_to_base(global_xyt, base_state)

            def obstacle_fn():
                return self.cam.is_obstacle_in_front()

            status = goto(self, list(base_xyt), dryrun=False, obstacle_fn=obstacle_fn)
            self._done = True
        return status

    def go_to_relative(self, xyt_position):
        """Moves the robot base to the given goal state relative to its current
        pose.

        :param xyt_position: The  relative goal state of the form (x,y,yaw)
        """
        status = "SUCCEEDED"

        if self._done:
            self.initialize_cam()
            self._done = False

            def obstacle_fn():
                return safe_call(self.cam, is_obstacle_in_front)

            status = goto(self, list(xyt_position), dryrun=False, obstacle_fn=obstacle_fn)
            self._done = True
        return status

    def is_moving(self):
        return not self._done

    def stop(self):
        robot.stop()
        robot.push_command()

    def remove_runstop(self):
        if robot.pimu.status["runstop_event"]:
            robot.pimu.runstop_event_reset()
            robot.push_command()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pass in server device IP")
    parser.add_argument(
        "--ip",
        help="Server device (robot) IP. Default is 192.168.0.0",
        type=str,
        default="0.0.0.0",
    )

    args = parser.parse_args()

    np.random.seed(123)

    with Pyro4.Daemon(args.ip) as daemon:
        robot = RemoteHelloRobot(ip=args.ip)
        robot_uri = daemon.register(robot)
        with Pyro4.locateNS() as ns:
            ns.register("hello_robot", robot_uri)

        print("Hello Robot Server is started...")
        daemon.requestLoop()