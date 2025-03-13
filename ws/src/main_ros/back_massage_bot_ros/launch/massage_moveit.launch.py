# Copyright (c) 2025, Gary Lvov, Vinay Balaji, Tim Bennet, Xandar Ingare, Ben Yoon
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            "arm_group_name", default_value="arm", description="Name of the MoveIt group for the robot arm"
        ),
        DeclareLaunchArgument(
            "end_effector_link", default_value="j2n6s300_end_effector", description="Name of the end effector link"
        ),
        # Launch the massage_moveit node
        Node(
            package="back_massage_bot_ros",
            executable="massage_moveit_node",
            name="massage_moveit",
            output="screen",
            parameters=[{
                "arm_group_name": LaunchConfiguration("arm_group_name"),
                "end_effector_link": LaunchConfiguration("end_effector_link"),
            }],
        ),
    ])
