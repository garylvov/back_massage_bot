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
    # Define the Massage Gun communication node
    massage_node = Node(
        package="back_massage_bot_ros",
        executable="massage_handler.py",
        name="massage_handler",
        output="screen",
        parameters=[
            # {"serial_port": "/dev/ttyUSB0"},  # Update with the correct serial port if needed
            # {"baud_rate": 9600},
            # {"input_topic": "esp32_logs"},
        ],
    )

    # Return the launch description with the Massage Gun node
    return LaunchDescription([massage_node])
