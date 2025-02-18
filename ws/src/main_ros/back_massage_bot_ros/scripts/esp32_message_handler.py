#!/usr/bin/env python3

# Copyright (c) 2025, Gary Lvov, Vinay Balaji, Tim Bennet, Xandar Ingare, Ben Yoon
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import re

import serial
import synchros2.process as ros_process
import synchros2.scope as ros_scope
from commands import commands
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from serial.tools import list_ports
from std_msgs.msg import String


# Function to remove ANSI escape codes from a string
def remove_ansi_escape_codes(text):
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)


class ESP32MessageHandler:
    def __init__(self, serial_port: str, baud_rate: int, output_topic: str):
        self.node = ros_scope.node()

        # Open the serial port
        self.serial_conn = serial.Serial(serial_port, baud_rate, timeout=1)

        # Create a publisher for ESP32 log messages
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.publisher = self.node.create_publisher(String, output_topic, qos)
        self.node.get_logger().info(f"Subscribing to serial port: {serial_port}")
        self.node.get_logger().info(f"Publishing to topic: {output_topic}")

        # Timer to read serial messages (2 Hz = every 0.5s)
        self.timer = self.node.create_timer(0.5, self.get_message)

    def get_message(self):
        """Reads a line from the serial port and publishes relevant logs."""
        # self.get_logger().info("Checking for messages...")
        # self.get_logger().info(f"Bytes in buffer: {self.serial_conn.in_waiting}")
        while self.serial_conn.in_waiting > 0:
            try:
                raw_data = self.serial_conn.readline().decode("utf-8").strip()
                raw_data = remove_ansi_escape_codes(raw_data)
                # self.node.get_logger().info(f"Received: {raw_data}")
                for command in commands:
                    if command in raw_data:
                        msg = String()
                        msg.data = command
                        self.publisher.publish(msg)
                        self.node.get_logger().info(f"Published: {msg.data}")
            except Exception as e:
                self.node.get_logger().error(f"Error reading from serial: {str(e)}")


@ros_process.main()
def main(args=None):
    # Find the correct serial port using serial.tools.list_ports
    connected_ports = list_ports.comports()
    serial_port = None
    for port in connected_ports:
        if "VID:PID=303A:1001" in port.hwid:
            print(port.device, port.description)
            serial_port = port.device
            break
    if not serial_port:
        print("Could not find the ESP32 serial port")
        return
    baud_rate = 9600
    output_topic = "esp32_logs"

    ESP32MessageHandler(serial_port, baud_rate, output_topic)
    main.wait_for_shutdown()


if __name__ == "__main__":
    main()
