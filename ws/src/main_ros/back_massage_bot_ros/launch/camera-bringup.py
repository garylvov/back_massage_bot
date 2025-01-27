# Copyright (c) 2025, Gary Lvov, Vinay Balaji, Tim Bennet, Xandar Ingare, Ben Yoon
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Define the RealSense camera node
    realsense_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="realsense_camera",
        output="screen",
        parameters=[
            {"enable_color": True},
            {"enable_depth": True},
            {"enable_infra1": True},
            {"enable_infra2": True},
            {"align_depth": True},  # Align depth to color stream
            {"pointcloud.enable": False},  # Enable point cloud if needed
            {"json_file_path": ""},  # Path to a custom JSON configuration file if required
        ],
    )

    # Return the launch description with the RealSense node
    return LaunchDescription([realsense_node])
