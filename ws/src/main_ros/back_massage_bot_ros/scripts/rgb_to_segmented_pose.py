#!/usr/bin/env python3

# Copyright (c) 2025, Gary Lvov, Vinay Balaji, Tim Bennet, Xandar Ingare, Ben Yoon
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import argparse
import colorsys
import queue
from collections import deque
from contextlib import closing
from threading import Lock

import cv2
import numpy as np
import open3d as o3d
import rclpy
import synchros2.process as ros_process
import synchros2.scope as ros_scope
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from synchros2.subscription import Subscription
from visualization_msgs.msg import Marker, MarkerArray

from back_massage_bot import get_pose_mask, rgb_to_segmented_pose_model
from back_massage_bot.utils import AreaColorsMapping, get_close_color_mask

# Default topic names for RealSense cameras
DEFAULT_DEPTH_IMAGE_TOPIC = "/depth/image_rect_raw"
DEFAULT_DEPTH_INFO_TOPIC = "/depth/camera_info"
DEFAULT_MARKER_TOPIC = "/visualization_marker_array"


def create_point_cloud_from_depth(depth_image, camera_info, mask=None, downsample_factor=1):
    """
    Create a point cloud from a depth image and camera info using Open3D.

    Args:
        depth_image: Depth image (numpy array)
        camera_info: Camera info message
        mask: Optional binary mask to filter points (numpy array)
        downsample_factor: Factor to downsample the point cloud (default: no downsampling)

    Returns:
        o3d.geometry.PointCloud: Open3D point cloud
    """
    # Extract camera parameters
    fx = camera_info.k[0]  # Focal length x
    fy = camera_info.k[4]  # Focal length y
    cx = camera_info.k[2]  # Principal point x
    cy = camera_info.k[5]  # Principal point y

    # Apply mask to depth image if provided
    if mask is not None:
        # Make sure mask and depth image have the same dimensions
        if mask.shape != depth_image.shape:
            # Convert boolean mask to uint8 before resizing (0 or 255)
            if mask.dtype == bool:
                mask_uint8 = np.uint8(mask) * 255
            else:
                mask_uint8 = mask

            # Resize the uint8 mask
            mask_uint8 = cv2.resize(
                mask_uint8, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_NEAREST
            )

            # Convert back to boolean mask
            mask = mask_uint8 > 0

        # Create a copy of the depth image
        masked_depth = depth_image.copy()

        # Set depth to 0 for pixels outside the mask
        masked_depth[~mask] = 0
    else:
        masked_depth = depth_image

    # Get image dimensions
    height, width = masked_depth.shape

    # Convert depth image to o3d format (must be float32 and scaled to meters)
    o3d_depth = o3d.geometry.Image(masked_depth.astype(np.float32) / 1000.0)  # Convert mm to meters

    # Create Open3D camera intrinsic
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

    # Create point cloud from depth image
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_depth,
        intrinsic,
        depth_scale=1.0,  # Already scaled to meters
        depth_trunc=10.0,  # Max depth in meters
        project_valid_depth_only=True,
    )

    return pcd


def create_marker_from_point_cloud(pcd, frame_id, marker_id, color_rgb, marker_type=Marker.POINTS, scale=0.01):
    """
    Create a ROS marker from an Open3D point cloud.

    Args:
        pcd: Open3D point cloud
        frame_id: Frame ID for the marker
        marker_id: Unique ID for the marker
        color_rgb: RGB color array [r, g, b] with values 0-255
        marker_type: Type of marker (default: POINTS)
        scale: Scale of the marker points

    Returns:
        visualization_msgs.msg.Marker: ROS marker
    """
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = ros_scope.node().get_clock().now().to_msg()
    marker.ns = "massage_areas"
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD

    # Set scale
    marker.scale.x = scale
    marker.scale.y = scale
    marker.scale.z = scale

    # Set color (convert from 0-255 to 0-1 range)
    # Extract a simple RGB color from the complex color structure
    # Just use the first color in the list for visualization
    if isinstance(color_rgb, list) and len(color_rgb) > 0:
        # If it's a list of tuples of tuples, extract the first RGB tuple
        if isinstance(color_rgb[0], tuple) and len(color_rgb[0]) > 0:
            first_color = color_rgb[0]
            if isinstance(first_color[0], tuple) and len(first_color[0]) == 3:
                r, g, b = first_color[0]
            else:
                # Default color if structure is unexpected
                r, g, b = 255, 0, 0  # Red
        else:
            # Default color if structure is unexpected
            r, g, b = 255, 0, 0  # Red
    else:
        # Default color if structure is unexpected
        r, g, b = 255, 0, 0  # Red

    marker.color = ColorRGBA(r=float(r) / 255.0, g=float(g) / 255.0, b=float(b) / 255.0, a=1.0)

    # Add points
    points_np = np.asarray(pcd.points)
    for point in points_np:
        p = Point()
        p.x = point[0]
        p.y = point[1]
        p.z = point[2]
        marker.points.append(p)

    return marker


def filter_largest_cluster(pcd, eps=0.001, min_points=20, outlier_radius=0.1, nb_points=30):
    """
    Apply DBSCAN clustering to a point cloud and return only the largest cluster
    with additional statistical outlier removal.

    Args:
        pcd: Open3D point cloud
        eps: DBSCAN epsilon parameter (cluster radius) - higher = more inclusive clusters
        min_points: Minimum points for a cluster - higher = more filtering
        outlier_radius: Radius for statistical outlier removal
        nb_points: Number of points to use for outlier statistics

    Returns:
        o3d.geometry.PointCloud: Filtered point cloud containing only the largest cluster
    """
    # Skip if point cloud is empty
    if len(pcd.points) == 0:
        return pcd

    # First, apply statistical outlier removal to clean up the point cloud
    try:
        # More aggressive first pass outlier removal
        # Lower std_ratio means more points will be considered outliers
        cleaned_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_points, std_ratio=0.3)  # Reduced from 2.0

        # If the cleaned point cloud is too small, use the original
        if len(cleaned_pcd.points) < 10:
            cleaned_pcd = pcd
    except Exception:
        # If outlier removal fails, use the original point cloud
        cleaned_pcd = pcd

    # Apply DBSCAN clustering with more aggressive parameters
    labels = np.array(cleaned_pcd.cluster_dbscan(eps=eps, min_points=min_points))

    # If no clusters found, return empty point cloud
    if labels.max() < 0:
        return o3d.geometry.PointCloud()

    # Count points in each cluster
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)

    # If no valid clusters, return empty point cloud
    if len(unique_labels) == 0:
        return o3d.geometry.PointCloud()

    # Find the largest cluster
    largest_cluster_label = unique_labels[np.argmax(counts)]

    # Create a new point cloud with only the largest cluster
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
    largest_cluster_pcd = cleaned_pcd.select_by_index(largest_cluster_indices)

    # Apply a second pass of outlier removal for even cleaner results
    try:
        # More aggressive second pass outlier removal
        # Lower std_ratio and higher nb_neighbors for more aggressive filtering
        final_pcd, _ = largest_cluster_pcd.remove_statistical_outlier(
            nb_neighbors=nb_points, std_ratio=1.0
        )  # Reduced from 1.5

        # If the final point cloud is too small, use the largest cluster without second filtering
        if len(final_pcd.points) < 10:
            final_pcd = largest_cluster_pcd
    except Exception:
        # If second outlier removal fails, use the largest cluster
        final_pcd = largest_cluster_pcd

    # Apply voxel downsampling to reduce point density
    try:
        voxel_size = 0.015  # Increased from 0.01 to 0.015 for more aggressive downsampling
        downsampled_pcd = final_pcd.voxel_down_sample(voxel_size)

        # If downsampling results in too few points, use the non-downsampled version
        if len(downsampled_pcd.points) < 10:
            return final_pcd
        return downsampled_pcd
    except Exception:
        # If downsampling fails, return the filtered point cloud
        return final_pcd


def visualize_clusters(pcd, frame_id, base_id, area_color, cluster_eps=0.05, min_points=10):
    """
    Create markers to visualize all clusters in a point cloud with different colors.

    Args:
        pcd: Open3D point cloud
        frame_id: Frame ID for the markers
        base_id: Base ID for the markers (will be incremented for each cluster)
        area_color: Base RGB color array for the area
        cluster_eps: DBSCAN epsilon parameter
        min_points: Minimum points for a cluster

    Returns:
        list: List of ROS markers, one for each cluster
    """
    # Skip if point cloud is empty
    if len(pcd.points) == 0:
        return []

    # Apply DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps=cluster_eps, min_points=min_points))

    # If no clusters found, return empty list
    if labels.max() < 0:
        return []

    # Get unique cluster labels (excluding noise points with label -1)
    unique_labels = np.unique(labels[labels >= 0])

    markers = []

    # Extract base color
    base_r, base_g, base_b = 255, 0, 0  # Default red
    if isinstance(area_color, list) and len(area_color) > 0:
        if isinstance(area_color[0], tuple) and len(area_color[0]) > 0:
            first_color = area_color[0]
            if isinstance(first_color[0], tuple) and len(first_color[0]) == 3:
                base_r, base_g, base_b = first_color[0]

    # Create a marker for each cluster with a color variation
    for i, label in enumerate(unique_labels):
        # Get points in this cluster
        cluster_indices = np.where(labels == label)[0]
        cluster_pcd = pcd.select_by_index(cluster_indices)

        # Skip if cluster is empty
        if len(cluster_pcd.points) == 0:
            continue

        # Create a color variation for this cluster
        # Largest cluster (i=0) gets the original color, others get variations
        if i == 0:
            # Largest cluster - use original color but brighter
            r, g, b = base_r, base_g, base_b
            scale = 0.015  # Larger points for main cluster
        else:
            # Other clusters - create color variations
            # Shift hue for different clusters
            hue_shift = (i * 30) % 360  # 30 degree hue shifts

            # Convert RGB to HSV, shift hue, convert back to RGB
            h, s, v = colorsys.rgb_to_hsv(base_r / 255.0, base_g / 255.0, base_b / 255.0)
            h = (h + hue_shift / 360.0) % 1.0
            r_float, g_float, b_float = colorsys.hsv_to_rgb(h, s, v)
            r, g, b = int(r_float * 255), int(g_float * 255), int(b_float * 255)

            scale = 0.01  # Normal size for other clusters

        # Create marker with the varied color
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = ros_scope.node().get_clock().now().to_msg()
        marker.ns = "massage_clusters"
        marker.id = base_id + i
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # Set scale
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale

        # Set color
        marker.color = ColorRGBA(r=float(r) / 255.0, g=float(g) / 255.0, b=float(b) / 255.0, a=1.0)

        # Add points
        points_np = np.asarray(cluster_pcd.points)
        for point in points_np:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]
            marker.points.append(p)

        # Add text to indicate cluster size
        marker.text = f"Cluster {i}: {len(cluster_indices)} points"

        markers.append(marker)

    return markers


class ImageProcessor:
    def __init__(
        self,
        input_topic: str,
        output_topic: str,
        depth_topic: str = DEFAULT_DEPTH_IMAGE_TOPIC,
        depth_info_topic: str = DEFAULT_DEPTH_INFO_TOPIC,
        marker_topic: str = DEFAULT_MARKER_TOPIC,
    ):
        self.node = ros_scope.node()
        self.bridge = CvBridge()
        (self.pose_predictor, self.pose_visualizer, self.pose_extractor) = rgb_to_segmented_pose_model.initialize_model(
            min_score=0.5
        )

        # Create queues for processing
        self.to_infer = queue.Queue(maxsize=1)  # Only keep latest image for inference
        self.num_frames_to_average = 20
        self.to_average = deque(maxlen=self.num_frames_to_average)  # Keep last processed images for averaging

        # Add depth image queue
        self.depth_queue = queue.Queue(maxsize=1)  # Only keep latest depth image
        self.depth_topic = depth_topic
        self.depth_info_topic = depth_info_topic
        self.marker_topic = marker_topic

        # Use Subscription for camera info instead of manual queue
        self.depth_info_sub = None
        if depth_info_topic:
            self.depth_info_sub = Subscription(CameraInfo, depth_info_topic, history_length=1)
            self.node.get_logger().info(f"Subscribing to depth camera info: {depth_info_topic}")

        # Add separate locks for inference and averaging
        self.inference_lock = Lock()
        self.averaging_lock = Lock()
        self.depth_lock = Lock()
        self.processing = False

        # QoS profile to only keep latest message
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1)

        # Create publisher and subscriber
        self.inferred_image_publisher = self.node.create_publisher(Image, output_topic, qos)
        self.sub = self.node.create_subscription(Image, input_topic, self.image_callback, qos)

        # Create depth subscriber if depth topic is provided
        if depth_topic:
            self.depth_sub = self.node.create_subscription(Image, depth_topic, self.depth_callback, qos)
            self.node.get_logger().info(f"Subscribing to depth: {depth_topic}")

        # Create marker publisher for visualization
        self.marker_publisher = self.node.create_publisher(MarkerArray, marker_topic, 10)

        # Create timer for processing
        self.inference_timer = self.node.create_timer(0.05, self.process_image)  # 0.05 seconds = 20 Hz
        self.average_timer = self.node.create_timer(0.5, self.plan_potential_massage)

        self.node.get_logger().info(f"Subscribing to: {input_topic}")
        self.node.get_logger().info(f"Publishing to: {output_topic}")
        self.node.get_logger().info(f"Publishing markers to: {marker_topic}")

        self.desired_areas = [AreaColorsMapping.BACK, AreaColorsMapping.LEGS]
        self.area_publishers = [
            self.node.create_publisher(Image, f"{output_topic}/{area.name}", qos) for area in self.desired_areas
        ]

    def image_callback(self, msg: Image) -> None:
        """Store the latest image in the inference queue"""
        with self.inference_lock:
            # Thread-safe approach to update the queue
            try:
                # Always try to put the new message
                if not self.to_infer.empty():
                    # Clear the queue if it has an item
                    try:
                        self.to_infer.get_nowait()
                    except queue.Empty:
                        pass  # Queue was emptied by another thread

                # Now put the new message
                self.to_infer.put_nowait(msg)
            except Exception as e:
                self.node.get_logger().error(f"Error in image callback: {e}")

    def depth_callback(self, msg: Image) -> None:
        """Store the latest depth image"""
        with self.depth_lock:
            try:
                # Always try to put the new message
                if not self.depth_queue.empty():
                    # Clear the queue if it has an item
                    try:
                        self.depth_queue.get_nowait()
                    except queue.Empty:
                        pass  # Queue was emptied by another thread

                # Now put the new message
                self.depth_queue.put_nowait(msg)
            except Exception as e:
                self.node.get_logger().error(f"Error in depth callback: {e}")

    def publish_point_cloud_markers(self, point_clouds, frame_id):
        """
        Publish only the largest cluster for each area as marker arrays.

        Args:
            point_clouds: Dictionary mapping area to point cloud (already filtered to largest cluster)
            frame_id: Frame ID for the markers
        """
        marker_array = MarkerArray()

        # First, add a deletion marker to clear all previous markers
        deletion_marker = Marker()
        deletion_marker.header.frame_id = frame_id
        deletion_marker.header.stamp = ros_scope.node().get_clock().now().to_msg()
        deletion_marker.ns = "largest_clusters"
        deletion_marker.id = 0
        deletion_marker.action = Marker.DELETEALL  # Delete all markers in this namespace
        marker_array.markers.append(deletion_marker)

        # Base ID for markers - start at 1 since 0 is used for deletion
        base_id = 1

        # Process each area
        for area, filtered_pcd in point_clouds.items():
            # Skip empty point clouds
            if len(filtered_pcd.points) == 0:
                continue

            # Get color for this area
            area_color = area.value

            try:
                # Create marker for the largest cluster only
                marker = create_marker_from_point_cloud(filtered_pcd, frame_id, base_id, area_color, scale=0.015)
                marker.ns = "largest_clusters"

                # Add text to indicate area name and point count
                marker.text = f"{area.name}: {len(filtered_pcd.points)} points"

                # Set a lifetime for the marker (2 seconds)
                marker.lifetime = rclpy.duration.Duration(seconds=2.0).to_msg()

                marker_array.markers.append(marker)
                base_id += 1

                self.node.get_logger().info(
                    f"Added marker for largest cluster of {area.name} with {len(filtered_pcd.points)} points"
                )
            except Exception as e:
                self.node.get_logger().error(f"Error creating marker for {area.name}: {e}")

        # Publish all markers under one topic
        self.marker_publisher.publish(marker_array)
        self.node.get_logger().info(f"Published {len(marker_array.markers)} markers in frame: {frame_id}")

    def plan_potential_massage(self):
        """Process and publish averaged area masks with depth information"""
        frames_to_average = None
        latest_depth_image = None
        latest_depth_info = None

        # Get the latest depth image if available
        with self.depth_lock:
            try:
                if not self.depth_queue.empty():
                    depth_msg = self.depth_queue.get()
                    latest_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                    self.node.get_logger().info(f"Got depth image with shape: {latest_depth_image.shape}")
            except Exception as e:
                self.node.get_logger().error(f"Error getting depth image: {e}")

        # Get the latest depth camera info if available
        if self.depth_info_sub and self.depth_info_sub.history:
            try:
                latest_depth_info = self.depth_info_sub.history[-1]  # Get the most recent message
                self.node.get_logger().info(f"Got depth camera info with frame_id: {latest_depth_info.header.frame_id}")
            except Exception as e:
                self.node.get_logger().error(f"Error getting depth camera info: {e}")

        with self.averaging_lock:
            if len(self.to_average) < self.num_frames_to_average:
                return  # Not enough frames to average

            # Make a copy of the queue to work with
            frames_to_average = list(self.to_average)

        if not frames_to_average:
            return

        try:
            # Use the most recent header
            latest_header = frames_to_average[-1][0]

            # Extract just the images
            images = [frame[1] for frame in frames_to_average]

            # Dictionary to store point clouds for each area
            area_point_clouds = {}

            # Process each desired area
            for i, area in enumerate(self.desired_areas):
                try:
                    # Extract masks for this area from each frame
                    area_masks = []
                    for img in images:
                        # Get mask for this specific area
                        # The area.value needs to be properly formatted for color comparison
                        color_value = np.array(area.value)

                        # Handle different possible shapes
                        if color_value.ndim == 3:  # If it's a 3D array like (1,2,3)
                            if color_value.shape[0] == 1:
                                # Take the first item if it's a 3D array with first dim of 1
                                color_value = color_value[0]

                        # If it's still 2D, take the first row
                        if color_value.ndim == 2:
                            color_value = color_value[0]

                        # Final check to ensure we have a 1D array
                        if color_value.ndim != 1:
                            self.node.get_logger().error(
                                f"Could not reduce color value to 1D array: {color_value.shape}"
                            )
                            continue

                        # Use a more permissive threshold for better detection
                        area_mask = get_close_color_mask(img, color_value, threshold=40)  # Increased from 40 to 60
                        area_masks.append(area_mask)

                    # Average the masks
                    if area_masks:
                        avg_mask = np.zeros_like(area_masks[0], dtype=float)
                        for mask in area_masks:
                            avg_mask += mask.astype(float)
                        avg_mask = (avg_mask / len(area_masks)).astype("uint8")

                        # Apply binary threshold to create a clean mask
                        threshold_value = int(255 * 0.4)  # 40% threshold
                        _, binary_mask = cv2.threshold(avg_mask, threshold_value, 255, cv2.THRESH_BINARY)

                        # Remove all morphological operations
                        # No closing, no dilation, no erosion

                        # If we have depth information, create point cloud directly from the mask
                        if latest_depth_image is not None and latest_depth_info is not None:
                            try:
                                # Convert binary mask to boolean mask
                                bool_mask = binary_mask > 0

                                # Create point cloud
                                pcd = create_point_cloud_from_depth(
                                    latest_depth_image, latest_depth_info, mask=bool_mask
                                )

                                # Apply more aggressive filtering parameters based on the area
                                if area == AreaColorsMapping.BACK:
                                    # Back needs larger clusters and more aggressive filtering
                                    filtered_pcd = filter_largest_cluster(
                                        pcd, eps=0.06, min_points=30, outlier_radius=0.12, nb_points=40
                                    )
                                elif area == AreaColorsMapping.LEGS:
                                    # Legs can use slightly less aggressive filtering
                                    filtered_pcd = filter_largest_cluster(
                                        pcd, eps=0.05, min_points=25, outlier_radius=0.1, nb_points=35
                                    )
                                else:
                                    # Default parameters for other areas
                                    filtered_pcd = filter_largest_cluster(
                                        pcd, eps=0.05, min_points=20, outlier_radius=0.1, nb_points=30
                                    )

                                # Store filtered point cloud for this area
                                area_point_clouds[area] = filtered_pcd

                                self.node.get_logger().info(
                                    f"Created point cloud for {area.name} with {len(pcd.points)} points, "
                                    f"filtered to {len(filtered_pcd.points)} points"
                                )
                            except Exception as e:
                                self.node.get_logger().error(f"Error creating point cloud: {e}")

                        # Publish the binary mask
                        mask_msg = self.bridge.cv2_to_imgmsg(binary_mask, encoding="mono8")
                        mask_msg.header = latest_header
                        self.area_publishers[i].publish(mask_msg)
                except Exception as e:
                    self.node.get_logger().error(f"Error processing area {area.name}: {e}")
                    # Log more details about the shapes
                    self.node.get_logger().error(
                        f"Area value shape: {np.array(area.value).shape}, Image shape:"
                        f" {images[0].shape if images else 'No images'}"
                    )

            # Publish point cloud markers if we have any
            if area_point_clouds and latest_depth_info:
                try:
                    frame_id = latest_depth_info.header.frame_id
                    self.node.get_logger().info(f"Publishing point cloud markers in frame: {frame_id}")
                    self.publish_point_cloud_markers(area_point_clouds, frame_id)
                except Exception as e:
                    self.node.get_logger().error(f"Error publishing point cloud markers: {e}")

        except Exception as e:
            self.node.get_logger().error(f"Error in publish_averaged_areas: {e}")

    def process_image(self):
        """Timer callback to process images"""
        # Check if we're already processing
        with self.inference_lock:
            if self.processing:
                return
            self.processing = True

        try:
            # Get latest image from inference queue
            latest_image = None
            with self.inference_lock:
                try:
                    latest_image = self.to_infer.get_nowait()
                except queue.Empty:
                    self.processing = False
                    return  # No image to process

            # Convert to CV2 image
            cv_image = self.bridge.imgmsg_to_cv2(latest_image, desired_encoding="passthrough")

            try:
                processed_image = rgb_to_segmented_pose_model.get_pose_mask(
                    cv_image, self.pose_predictor, self.pose_visualizer, self.pose_extractor
                )

                # Add to averaging queue with appropriate lock
                with self.averaging_lock:
                    self.to_average.append((latest_image.header, processed_image))

            except ValueError as e:
                self.node.get_logger().error(f"Error processing image: {e}")
                with self.inference_lock:
                    self.processing = False
                return

            # Convert back to ROS message and publish
            new_msg = self.bridge.cv2_to_imgmsg(processed_image, encoding="rgb8")
            new_msg.header = latest_image.header  # Preserve header
            self.inferred_image_publisher.publish(new_msg)

        except Exception as e:
            self.node.get_logger().error(f"Error processing image: {e}")
        finally:
            with self.inference_lock:
                self.processing = False

    def __del__(self):
        # Clean up subscription resources
        if self.depth_info_sub:
            self.depth_info_sub.close()


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process and republish RGB images")
    parser.add_argument("--input-topic", "-i", required=True, help="Input image topic to subscribe to")
    parser.add_argument("--output-topic", "-o", required=True, help="Output topic to publish processed images")
    parser.add_argument(
        "--depth-topic",
        "-d",
        default=DEFAULT_DEPTH_IMAGE_TOPIC,
        help=f"Depth image topic to subscribe to (default: {DEFAULT_DEPTH_IMAGE_TOPIC})",
    )
    parser.add_argument(
        "--depth-info-topic",
        "-c",
        default=DEFAULT_DEPTH_INFO_TOPIC,
        help=f"Depth camera info topic (default: {DEFAULT_DEPTH_INFO_TOPIC})",
    )
    parser.add_argument(
        "--marker-topic",
        "-m",
        default=DEFAULT_MARKER_TOPIC,
        help=f"Topic to publish visualization markers (default: {DEFAULT_MARKER_TOPIC})",
    )
    return parser


@ros_process.main(cli())
def main(args: argparse.Namespace) -> None:
    ImageProcessor(
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        depth_topic=args.depth_topic,
        depth_info_topic=args.depth_info_topic,
        marker_topic=args.marker_topic,
    )
    main.wait_for_shutdown()


if __name__ == "__main__":
    main()
