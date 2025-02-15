# Copyright (c) 2025, Gary Lvov, Vinay Balaji, Tim Bennet, Xandar Ingare, Ben Yoon
# All rights reserved.
#
# SPDX-License-Identifier: MIT

# Copyright (c) 2025, Gary Lvov, Vinay Balaji, Tim Bennett, Xandar Ingare, Ben Yoon
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Define the ESP32 communication node
    esp32_node = Node(
        package="esp32_s3_box",
        executable="esp32_node",
        name="esp32",
        output="screen",
        parameters=[
            # {"serial_port": "/dev/ttyUSB0"},
        ],
    )

    # Return the launch description with the RealSense node
    return LaunchDescription([esp32_node])
