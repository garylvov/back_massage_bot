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
    # Define the plan massage node
    plan_massage_node = Node(
        package="back_massage_bot_ros",
        executable="plan_massage.py",
        name="point_cloud_transformer_and_occupancy_mapper",
        output="screen",
        parameters=[
            # {"serial_port": "/dev/ttyUSB0"},  # Update with the correct serial port if needed
            # {"baud_rate": 9600},
            # {"output_topic": "esp32_logs"},
        ],
    )

    # Return the launch description with the ESP32 node
    return LaunchDescription([plan_massage_node])
