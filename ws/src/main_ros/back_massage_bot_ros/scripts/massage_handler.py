#!/usr/bin/env python3

# Copyright (c) 2025, Gary Lvov, Vinay Balaji, Tim Bennet, Xandar Ingare, Ben Yoon
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import re

import serial
import synchros2.process as ros_process
import synchros2.scope as ros_scope
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from serial.tools import list_ports
from std_msgs.msg import String


# Function to remove ANSI escape codes from a string
def remove_ansi_escape_codes(text):
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)


class MassageHandler:
    def __init__(self, serial_port: str, baud_rate: int, input_topic: str):
        self.node = ros_scope.node()

        # Open the serial port
        self.serial_conn = serial.Serial(serial_port, baud_rate, timeout=0, write_timeout=0, inter_byte_timeout=None)
        # Open and reset buffers
        self.serial_conn.reset_input_buffer()
        self.serial_conn.reset_output_buffer()

        # Create a subscriber for ESP32 log messages
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            avoid_ros_namespace_conventions=True,
            depth=1,
        )
        self.subscriber = self.node.create_subscription(String, input_topic, self.send_message, qos)
        self.node.get_logger().info(f"Subscribing to topic: {input_topic}")
        self.node.get_logger().info(f"Publishing to serial port: {serial_port}")

    def send_message(self, msg: String):
        """Writes the message from the input topic to the serial port."""
        try:
            message = msg.data + "\r\n"
            self.serial_conn.write(message.encode("utf-8"))
            self.serial_conn.flush()  # Add flush to ensure immediate sending
            self.node.get_logger().info(f"Sent: {msg.data}")
        except Exception as e:
            self.node.get_logger().error(f"Error writing to serial: {str(e)}")


@ros_process.main()
def main(args=None):
    # Find the correct serial port using serial.tools.list_ports
    connected_ports = list_ports.comports()
    serial_port = None
    for port in connected_ports:
        if "VID:PID=2341:0043" in port.hwid:
            print(port.device, port.description)
            serial_port = port.device
            break
    if not serial_port:
        print("Could Not find the Massage Gun Serial Port")
        return
    baud_rate = 115200
    input_topic = "esp32_logs"

    MassageHandler(serial_port, baud_rate, input_topic)
    main.wait_for_shutdown()


if __name__ == "__main__":
    main()
