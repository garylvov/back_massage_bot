#!/usr/bin/env python3
# Copyright (c) 2025, Gary Lvov, Vinay Balaji, Tim Bennet, Xandar Ingare, Ben Yoon
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
A CLI tool to send relative Cartesian commands to the robot arm.

Usage:
  relative_cartesian_command --dx 0.1 --dy 0.05 --dz 0.0
"""

import argparse
import sys
import time

import synchros2.process as ros_process
from geometry_msgs.msg import PoseStamped
from synchros2.utilities import namespace_with


@ros_process.main(uses_tf=True)
def main() -> int:
    """Main function for the relative cartesian command CLI."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Send relative Cartesian commands to the robot arm")
    parser.add_argument("--dx", type=float, default=0.0, help="Relative movement in X direction (meters)")
    parser.add_argument("--dy", type=float, default=0.0, help="Relative movement in Y direction (meters)")
    parser.add_argument("--dz", type=float, default=0.0, help="Relative movement in Z direction (meters)")
    parser.add_argument("--droll", type=float, default=0.0, help="Relative roll rotation (radians)")
    parser.add_argument("--dpitch", type=float, default=0.0, help="Relative pitch rotation (radians)")
    parser.add_argument("--dyaw", type=float, default=0.0, help="Relative yaw rotation (radians)")
    parser.add_argument("--base-frame", type=str, default="j2n6s300_link_base", help="Base frame for transformation")
    parser.add_argument("--ee-frame", type=str, default="j2n6s300_end_effector", help="End effector frame")
    parser.add_argument("--tf-prefix", type=str, default=None, help="TF prefix for frame names")

    args = parser.parse_args()

    # Get the node and create publisher
    node = main.node
    publisher = node.create_publisher(PoseStamped, "/arm_dispatch_command", 10)

    # Get the TF listener
    tf_listener = main.tf_listener
    if tf_listener is None:
        node.get_logger().error("Failed to get TF listener")
        return 1

    # Apply namespace prefix if provided
    base_frame = args.base_frame
    end_effector_frame = args.ee_frame
    if args.tf_prefix:
        base_frame = namespace_with(args.tf_prefix, base_frame)
        end_effector_frame = namespace_with(args.tf_prefix, end_effector_frame)

    # Get current pose
    try:
        node.get_logger().info(f"Looking up transform from {base_frame} to {end_effector_frame}")

        # Wait a moment for TF to be available
        time.sleep(0.5)

        transform = tf_listener.lookup_a_tform_b(base_frame, end_effector_frame, timeout_sec=2.0, wait_for_frames=True)

        # Create a PoseStamped from the transform
        current_pose = PoseStamped()
        current_pose.header.frame_id = base_frame
        current_pose.header.stamp = node.get_clock().now().to_msg()

        # Set position from transform
        current_pose.pose.position.x = transform.transform.translation.x
        current_pose.pose.position.y = transform.transform.translation.y
        current_pose.pose.position.z = transform.transform.translation.z

        # Set orientation from transform
        current_pose.pose.orientation = transform.transform.rotation

        node.get_logger().info(
            f"Current pose: x={current_pose.pose.position.x:.3f}, "
            f"y={current_pose.pose.position.y:.3f}, "
            f"z={current_pose.pose.position.z:.3f}"
        )

    except Exception as e:
        node.get_logger().error(f"Failed to get current pose: {str(e)}")
        return 1

    # Create target pose by adding relative offsets
    target_pose = PoseStamped()
    target_pose.header = current_pose.header

    # Add position offsets
    target_pose.pose.position.x = current_pose.pose.position.x + args.dx
    target_pose.pose.position.y = current_pose.pose.position.y + args.dy
    target_pose.pose.position.z = current_pose.pose.position.z + args.dz

    # For simplicity, we'll keep the same orientation for now
    # In a more advanced version, we could apply rotation offsets
    target_pose.pose.orientation = current_pose.pose.orientation

    # Publish the command
    node.get_logger().info(f"Sending relative move command: dx={args.dx}, dy={args.dy}, dz={args.dz}")
    node.get_logger().info(
        f"Target pose: x={target_pose.pose.position.x:.3f}, "
        f"y={target_pose.pose.position.y:.3f}, "
        f"z={target_pose.pose.position.z:.3f}"
    )

    publisher.publish(target_pose)

    # Wait a moment to ensure the message is published
    time.sleep(0.5)

    return 0


if __name__ == "__main__":
    sys.exit(main())
