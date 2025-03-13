#!/usr/bin/env python3

# Copyright (c) 2025, Gary Lvov, Vinay Balaji, Tim Bennet, Xandar Ingare, Ben Yoon
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import sys

import moveit_commander
import rclpy
from geometry_msgs.msg import Pose
from moveit_commander.move_group_commander import MoveGroupCommander
from moveit_commander.planning_scene_interface import PlanningSceneInterface
from rclpy.node import Node


class ArmController(Node):
    def __init__(self):
        super().__init__("arm_controller")

        # Initialize moveit_commander
        moveit_commander.roscpp_initialize(sys.argv)

        # Initialize the robot commander
        self.robot = moveit_commander.RobotCommander()

        # Initialize the planning scene interface
        self.scene = PlanningSceneInterface()

        # Create move group interface
        self.move_group = MoveGroupCommander("arm")

        # Set parameters
        self.move_group.set_max_velocity_scaling_factor(0.1)
        self.move_group.set_max_acceleration_scaling_factor(0.1)
        self.move_group.set_planning_time(10.0)

    def move_to_pose(self, x, y, z):
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0

        self.move_group.set_pose_target(pose)

        # Plan and execute
        success = self.move_group.go(wait=True)  # noqa: R504
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        return success  # noqa: R504


def main():
    rclpy.init()
    controller = ArmController()

    # Example move
    controller.move_to_pose(0.4, 0.0, 0.6)

    rclpy.spin(controller)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
