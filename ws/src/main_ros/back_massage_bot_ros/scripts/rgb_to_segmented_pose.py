#!/usr/bin/env python3

# Copyright (c) 2025, Gary Lvov, Vinay Balaji, Tim Bennet, Xandar Ingare, Ben Yoon
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import argparse

import synchros2.process as ros_process
import synchros2.scope as ros_scope
from cv_bridge import CvBridge
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

from back_massage_bot import rgb_to_segmented_pose_model


class ImageProcessor:
    def __init__(self, input_topic: str, output_topic: str):
        self.node = ros_scope.node()
        self.bridge = CvBridge()
        (self.pose_predictor, self.pose_visualizer, self.pose_extractor) = (
            rgb_to_segmented_pose_model.initialize_model()
        )

        # Create a queue for the latest image
        self.latest_image = None
        self.processing = False

        # QoS profile to only keep latest message
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1)

        # Create publisher and subscriber
        self.pub = self.node.create_publisher(Image, output_topic, qos)
        self.sub = self.node.create_subscription(Image, input_topic, self.image_callback, qos)

        # Create timer for 2 Hz processing
        self.timer = self.node.create_timer(0.5, self.process_image)  # 0.5 seconds = 2 Hz

        self.node.get_logger().info(f"Subscribing to: {input_topic}")
        self.node.get_logger().info(f"Publishing to: {output_topic}")

    def image_callback(self, msg: Image) -> None:
        """Just store the latest image"""
        self.latest_image = msg

    def process_image(self):
        """Timer callback to process images at 2 Hz"""
        if self.latest_image is None or self.processing:
            return

        self.processing = True
        try:
            # Convert to CV2 image
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding="passthrough")

            try:
                cv_image = rgb_to_segmented_pose_model.get_pose_mask(
                    cv_image, self.pose_predictor, self.pose_visualizer, self.pose_extractor
                )
            except ValueError as e:
                self.node.get_logger().error(f"Error processing image: {e}")
                return

            # Convert back to ROS message and publish
            new_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="rgb8")
            new_msg.header = self.latest_image.header  # Preserve header
            self.pub.publish(new_msg)

        except Exception as e:
            self.node.get_logger().error(f"Error processing image: {e}")
        finally:
            self.processing = False


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process and republish RGB images")
    parser.add_argument("--input-topic", "-i", required=True, help="Input image topic to subscribe to")
    parser.add_argument("--output-topic", "-o", required=True, help="Output topic to publish processed images")
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> None:
    ImageProcessor(input_topic=args.input_topic, output_topic=args.output_topic)
    main.wait_for_shutdown()


if __name__ == "__main__":
    main()
