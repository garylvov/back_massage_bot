#!/usr/bin/env python3

# Copyright (c) 2025, Gary Lvov, Vinay Balaji, Tim Bennet, Xandar Ingare, Ben Yoon
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import argparse
import os
from datetime import datetime

import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2


class PointCloudToPLY(Node):
    def __init__(self, args):
        super().__init__("pointcloud_to_ply")

        # Use command line arguments instead of ROS parameters
        self.topic = args.topic
        self.output_dir = os.path.expanduser(args.output_dir)
        self.save_on_timer = args.save_on_timer
        self.timer_period = args.timer_period
        self.frame_limit = args.frame_limit

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Counter for saved frames
        self.saved_frames = 0

        # Create a subscription
        self.subscription = self.create_subscription(PointCloud2, self.topic, self.point_cloud_callback, 10)

        # Create a timer if save_on_timer is True
        if self.save_on_timer:
            self.timer = self.create_timer(self.timer_period, self.timer_callback)
            self.latest_cloud = None
            self.get_logger().info(f"PointCloud to PLY converter: Saving every {self.timer_period} seconds")
        else:
            self.get_logger().info("PointCloud to PLY converter: Saving next available frame")

    def point_cloud_callback(self, msg):
        if self.save_on_timer:
            # Store the latest cloud for timer to process
            self.latest_cloud = msg
        else:
            # Save immediately
            if self.saved_frames < self.frame_limit:
                self.save_point_cloud(msg)

    def timer_callback(self):
        if self.latest_cloud is not None and self.saved_frames < self.frame_limit:
            self.save_point_cloud(self.latest_cloud)
            self.latest_cloud = None

    def save_point_cloud(self, cloud_msg):
        try:
            # Convert PointCloud2 to points
            points = []
            for point in pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z")):
                points.append(point)

            # If there are no points, don't save
            if len(points) == 0:
                self.get_logger().warn("Received empty point cloud, not saving")
                return

            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pointcloud_{timestamp}.ply"
            filepath = os.path.join(self.output_dir, filename)

            # Convert to numpy array for easier manipulation
            points_array = np.array(points)

            # Write PLY file
            self.write_ply(filepath, points_array)

            self.saved_frames += 1
            self.get_logger().info(f"Saved point cloud to {filepath} ({len(points)} points)")

            # Shut down if we've reached the frame limit
            if self.saved_frames >= self.frame_limit and not self.save_on_timer:
                self.get_logger().info(f"Saved {self.frame_limit} frames. Shutting down.")
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error saving point cloud: {str(e)}")

    def write_ply(self, filepath, points):
        """Write points to a PLY file"""
        with open(filepath, "w") as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            # Write points (setting all to white since the original points are white)
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]} 255 255 255\n")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Save PointCloud2 messages to PLY files")
    parser.add_argument("--topic", type=str, default="/depth/color/points", help="PointCloud2 topic to subscribe to")
    parser.add_argument("--output-dir", type=str, default="~/pointcloud_plys", help="Directory to save PLY files")
    parser.add_argument("--save-on-timer", action="store_true", help="Save on a timer instead of immediately")
    parser.add_argument("--timer-period", type=float, default=5.0, help="Period (seconds) for timer-based saving")
    parser.add_argument("--frame-limit", type=int, default=1, help="Number of frames to save before shutting down")

    # Parse arguments
    args = parser.parse_args()

    # Initialize ROS
    rclpy.init()

    # Create and run node with parsed arguments
    node = PointCloudToPLY(args)
    rclpy.spin(node)

    # Cleanup
    rclpy.shutdown()


if __name__ == "__main__":
    main()
