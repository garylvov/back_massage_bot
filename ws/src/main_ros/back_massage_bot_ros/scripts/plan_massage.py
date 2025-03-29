#!/usr/bin/env python3

# Copyright (c) 2025, Gary Lvov, Tim Bennett, Xander Ingare, Ben Yoon, Vinay Balaji
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
A ROS2 node for transforming and cropping point clouds from camera to robot frame.
Subscribes to a point cloud topic, transforms points to the robot base frame using Open3D,
crops based on specified bounds, and creates an occupancy grid with 1cm resolution.
Runs YOLO inference directly on the occupancy grid to detect body regions.
"""

import argparse
import os
import sys
import numpy as np
import open3d as o3d
from datetime import datetime
import ultralytics
import cv2
import threading
import queue
import time

import synchros2.process as ros_process
import synchros2.scope as ros_scope
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped
from synchros2.context import wait_for_shutdown
from synchros2.utilities import namespace_with
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

# Import visualization utilities
from visualization_utils import (
    create_detection_debug_markers,
    create_detection_markers
)

class PointCloudTransformerAndOccupancyMapper:
    def __init__(
        self,
        input_topic: str = "/depth/color/points",
        output_topic: str = "/massage_planning/cropped_cloud",
        occupancy_topic: str = "/massage_planning/debug_visualization",
        camera_frame: str = "camera_depth_optical_frame",
        robot_base_frame: str = "j2n6s300_link_base",
        tf_prefix: str | None = None,
        x_min: float = -0.7,
        x_max: float = 1.4,
        y_min: float = -0.8,
        y_max: float = -0.1,
        z_min: float = 0.0,
        z_max: float = 1.0,
        grid_resolution: float = 0.01,  # 1cm grid resolution
        output_dir: str = "~/pointcloud_plys",
        save_plys: bool = False,
        save_grids: bool = False,
        model_path: str = "/back_massage_bot/src/back_massage_bot/models/third_runs_the_charm/weights/best.pt"
    ):
        # Get node and tf listener from current scope
        self.node = ros_scope.node()
        self.tf_listener = ros_scope.tf_listener()
        
        if self.tf_listener is None:
            self.node.get_logger().error("Failed to get TF listener")
            raise RuntimeError("TF listener not available")
        
        # Store configuration
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.occupancy_topic = occupancy_topic
        self.camera_frame = camera_frame
        self.robot_base_frame = robot_base_frame
        self.crop_bounds = {
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "z_min": z_min, "z_max": z_max
        }
        self.grid_resolution = grid_resolution
        self.save_plys = save_plys
        self.save_grids = save_grids
        self.output_dir = os.path.expanduser(output_dir)
        
        # Create a unique directory for this run if needed
        if self.save_plys or self.save_grids:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(self.output_dir, f"run_{timestamp}")
            os.makedirs(self.run_dir, exist_ok=True)
            self.node.get_logger().info(f"Created output directory at {self.run_dir}")
            
            if self.save_grids:
                self.grid_dir = os.path.join(self.run_dir, "grids")
                os.makedirs(self.grid_dir, exist_ok=True)
                self.node.get_logger().info(f"Will save grid images to {self.grid_dir}")
        
        # Apply namespace prefix if provided
        if tf_prefix:
            self.camera_frame = namespace_with(tf_prefix, self.camera_frame)
            self.robot_base_frame = namespace_with(tf_prefix, self.robot_base_frame)
        
        # Create QoS profile for point cloud
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)
        
        # Create publishers
        self.cloud_publisher = self.node.create_publisher(PointCloud2, self.output_topic, qos)
        self.marker_publisher = self.node.create_publisher(MarkerArray, self.occupancy_topic, qos)
        
        # Create CV bridge for image conversion
        self.cv_bridge = CvBridge()
        
        # Create image publishers for debugging
        self.yolo_input_pub = self.node.create_publisher(Image, "/massage_planning/yolo_input", qos)
        self.yolo_output_pub = self.node.create_publisher(Image, "/massage_planning/yolo_output", qos)
        self.grid_image_pub = self.node.create_publisher(Image, "/massage_planning/grid_image", qos)
        
        # Cache for the transform - only look it up once
        self.transform_matrix = None
        self.tried_transform_lookup = False
        
        # Frame counter
        self.frame_counter = 0
        
        # Create a queue for point cloud messages
        self.point_cloud_queue = queue.Queue(maxsize=10)
        
        # Start the processing thread
        self.processing_thread = threading.Thread(target=self.process_point_clouds, daemon=True)
        self.processing_thread.start()
        self.node.get_logger().info("Started point cloud processing thread")
        
        # Subscribe to the point cloud topic
        self.subscription = self.node.create_subscription(
            PointCloud2,
            self.input_topic,
            self.point_cloud_callback,
            qos
        )
        
        self.node.get_logger().info(f"Subscribed to {self.input_topic}")
        self.node.get_logger().info(f"Publishing transformed and cropped point cloud to {self.output_topic}")
        self.node.get_logger().info(f"Publishing visualization to {self.occupancy_topic}")
        self.node.get_logger().info("Waiting for point clouds...")

        # Initialize YOLO model for inference
        self.node.get_logger().info(f"Attempting to load YOLO model from {model_path}")
        if not os.path.exists(model_path):
            self.node.get_logger().error(f"Model file does not exist: {model_path}")
        
        try:
            self.model = ultralytics.YOLO(model_path)
            self.model.args['data'] = "data/data.yaml"
            self.node.get_logger().info(f"Successfully loaded YOLO model from {model_path}")
        except Exception as e:
            self.model = None
            self.node.get_logger().error(f"Failed to load YOLO model: {str(e)}")
            import traceback
            self.node.get_logger().error(traceback.format_exc())
        
        # Create additional publisher for detected regions
        self.regions_topic = "/massage_planning/body_regions"
        self.detection_publisher = self.node.create_publisher(MarkerArray, self.regions_topic, qos)
        self.node.get_logger().info(f"Will publish detected regions to {self.regions_topic}")

    def lookup_transform(self) -> bool:
        """Look up the transform between camera and robot base frame, return True if successful"""
        if self.transform_matrix is not None:
            return True
            
        if self.tried_transform_lookup:
            return False
            
        self.tried_transform_lookup = True
        
        try:
            # Look up the transform from robot base frame to camera frame (opposite direction)
            # This lets us transform points from camera frame to robot base frame
            transform = self.tf_listener.lookup_a_tform_b(
                self.robot_base_frame,  # target frame 
                self.camera_frame,      # source frame
                timeout_sec=2.0,
                wait_for_frames=True
            )
            
            # Extract transform components
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            
            # Convert ROS transform to 4x4 transformation matrix for Open3D
            quat = [rotation.x, rotation.y, rotation.z, rotation.w]
            rot_matrix = Rotation.from_quat(quat).as_matrix()
            
            # Create transformation matrix
            self.transform_matrix = np.eye(4)
            self.transform_matrix[:3, :3] = rot_matrix
            self.transform_matrix[:3, 3] = [translation.x, translation.y, translation.z]
            self.node.get_logger().info(f"Successfully cached transform from {self.camera_frame} to {self.robot_base_frame}")
            return True
            
        except Exception as e:
            self.node.get_logger().error(f"Failed to get transform: {str(e)}")
            return False

    def create_top_down_occupancy(self, points):
        """Create a 2D top-down occupancy grid from the point cloud"""
        try:
            # Get abounds for the grid
            x_min = self.crop_bounds["x_min"]
            x_max = self.crop_bounds["x_max"]
            y_min = self.crop_bounds["y_min"]
            y_max = self.crop_bounds["y_max"]
            
            # Calculate grid size
            grid_width = int(np.ceil((x_max - x_min) / self.grid_resolution))
            grid_height = int(np.ceil((y_max - y_min) / self.grid_resolution))
            
            # Create empty binary mask - this is our occupancy grid
            # IMPORTANT: This grid has (0,0) at the bottom-left in robot coordinates
            mask = np.zeros((grid_width, grid_height), dtype=bool)
            
            # Create array to store a representative point for each occupied cell
            representative_points = np.empty((grid_width, grid_height), dtype=object)
            for i in range(grid_width):
                for j in range(grid_height):
                    representative_points[i, j] = None
            
            # Fill grid cells
            for point in points:
                # Calculate grid cell indices
                x_idx = int((point[0] - x_min) / self.grid_resolution)
                y_idx = int((point[1] - y_min) / self.grid_resolution)
                
                # Check if the indices are within grid bounds
                if 0 <= x_idx < grid_width and 0 <= y_idx < grid_height:
                    mask[x_idx, y_idx] = True
                    # Store the first point we find for each cell (or could be replaced with criteria like lowest z)
                    if representative_points[x_idx, y_idx] is None:
                        representative_points[x_idx, y_idx] = point
            
            self.node.get_logger().info(f"Created binary mask with dimensions {grid_width}x{grid_height}")
            
            # Create an image for YOLO with the same dimensions as our grid
            # IMPORTANT: YOLO expects images with (0,0) at the top-left
            grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            # Log the mask shape and grid image shape
            self.node.get_logger().info(f"Mask shape: {mask.shape}, Grid image shape: {grid_image.shape}")
            
            # Fill the grid image
            # CRITICAL: We need to flip the y-axis when converting from our grid to the image
            # Our grid has (0,0) at bottom-left, but the image has (0,0) at top-left
            for i in range(grid_width):
                for j in range(grid_height):
                    if mask[i, j]:
                        # Flip y-axis for the image (j → grid_height-1-j)
                        grid_image[grid_height-1-j, i] = [255, 255, 255]  # White for occupied cells
            
            # Save grid image if requested
            if self.save_grids:
                grid_image_path = os.path.join(self.grid_dir, f"grid_{self.frame_counter:04d}.png")
                try:
                    import cv2
                    cv2.imwrite(grid_image_path, grid_image)
                    self.node.get_logger().info(f"Saved grid image to {grid_image_path}")
                except Exception as e:
                    self.node.get_logger().error(f"Failed to save grid image: {str(e)}")
            
            # Publish the grid image used for YOLO detection
            try:
                bridge = CvBridge()
                grid_img_msg = bridge.cv2_to_imgmsg(grid_image, encoding="rgb8")
                grid_img_msg.header.stamp = self.node.get_clock().now().to_msg()
                grid_img_msg.header.frame_id = self.robot_base_frame
                self.grid_image_pub.publish(grid_img_msg)
                self.node.get_logger().info("Published original grid image used for YOLO detection")
            except Exception as e:
                self.node.get_logger().error(f"Error publishing grid image: {str(e)}")
            
            return {
                "mask": mask,  # Binary mask for occupied/unoccupied cells
                "points": representative_points,  # Representative point for each cell
                "origin": (x_min, y_min),
                "image": grid_image,  # Image representation for YOLO
                "grid_dims": (grid_width, grid_height)  # Grid dimensions
            }
        except Exception as e:
            self.node.get_logger().error(f"Error creating occupancy grid: {str(e)}")
            return None

    def run_yolo_inference(self, grid_image):
        """Run YOLO inference directly on the grid image numpy array"""
        if self.model is None:
            self.node.get_logger().warn("YOLO model not available, skipping inference")
            return None
        
        try:
            # Get original image dimensions
            original_height, original_width = grid_image.shape[:2]
            self.node.get_logger().info(f"Original grid image dimensions: {original_width}x{original_height}")
            
            # Publish the input image for visualization
            input_img_msg = self.cv_bridge.cv2_to_imgmsg(grid_image, encoding="rgb8")
            input_img_msg.header.stamp = self.node.get_clock().now().to_msg()
            input_img_msg.header.frame_id = self.robot_base_frame
            self.yolo_input_pub.publish(input_img_msg)
            self.node.get_logger().info("Published YOLO input image")
            
            # Run inference directly on the numpy array
            self.node.get_logger().info(f"Running YOLO inference on grid image array")
            results = self.model(grid_image)
            
            # Check if we got any detections
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Store original image dimensions in the result for later use
                results[0].orig_shape = grid_image.shape[:2]  # (height, width)
                
                self.node.get_logger().info(f"YOLO detected {len(results[0].boxes)} objects")
                for i, box in enumerate(results[0].boxes):
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    xyxy = box.xyxy[0].cpu().numpy()
                    self.node.get_logger().info(f"Detection {i}: class={cls}, conf={conf:.2f}, box={xyxy}")
                
                # Create output image with detection boxes
                output_img = results[0].plot()
                output_img_msg = self.cv_bridge.cv2_to_imgmsg(output_img, encoding="rgb8")
                output_img_msg.header.stamp = self.node.get_clock().now().to_msg()
                output_img_msg.header.frame_id = self.robot_base_frame
                self.yolo_output_pub.publish(output_img_msg)
                self.node.get_logger().info("Published YOLO output image with detections")
                
                return results[0]
            else:
                self.node.get_logger().warn("YOLO detected no objects")
                
                # Try with lower confidence threshold
                self.node.get_logger().info("Trying YOLO with increased confidence threshold")
                results = self.model(grid_image, conf=0.1)  # Lower confidence threshold
                
                if len(results) > 0 and len(results[0].boxes) > 0:
                    # Store original image dimensions in the result for later use
                    results[0].orig_shape = grid_image.shape[:2]  # (height, width)
                    
                    self.node.get_logger().info(f"YOLO detected {len(results[0].boxes)} objects with lower threshold")
                    
                    # Create output image with detection boxes
                    output_img = results[0].plot()
                    output_img_msg = self.cv_bridge.cv2_to_imgmsg(output_img, encoding="rgb8")
                    output_img_msg.header.stamp = self.node.get_clock().now().to_msg()
                    output_img_msg.header.frame_id = self.robot_base_frame
                    self.yolo_output_pub.publish(output_img_msg)
                    self.node.get_logger().info("Published YOLO output image with detections (lower threshold)")
                    
                    return results[0]
                
                return None
        except Exception as e:
            self.node.get_logger().error(f"Error during YOLO inference: {str(e)}")
            import traceback
            self.node.get_logger().error(traceback.format_exc())
            return None

    def point_cloud_callback(self, msg: PointCloud2) -> None:
        """Callback for point cloud messages - just adds to queue"""
        try:
            # Add the message to the queue, with a timeout to avoid blocking
            self.point_cloud_queue.put(msg, block=True, timeout=0.1)
            self.node.get_logger().debug(f"Added point cloud to queue (size: {self.point_cloud_queue.qsize()})")
        except queue.Full:
            self.node.get_logger().warn("Point cloud queue is full, dropping message")
        except Exception as e:
            self.node.get_logger().error(f"Error in point cloud callback: {str(e)}")

    def process_point_clouds(self):
        """Process point clouds from the queue in a separate thread"""
        self.node.get_logger().info("Point cloud processing thread started")
        
        while True:
            try:
                # Get a message from the queue
                msg = self.point_cloud_queue.get(block=True)
                self.node.get_logger().debug("Processing point cloud from queue")
                
                # Process the point cloud
                self.process_point_cloud(msg)
                
                # Mark the task as done
                self.point_cloud_queue.task_done()
                
            except Exception as e:
                self.node.get_logger().error(f"Error in processing thread: {str(e)}")
                import traceback
                self.node.get_logger().error(traceback.format_exc())
                
                # Sleep briefly to avoid tight loop in case of persistent errors
                time.sleep(0.1)

    def process_point_cloud(self, msg: PointCloud2) -> None:
        """Process a single point cloud message using Open3D"""
        try:
            # Try to get the transform if we don't have it yet
            if not self.lookup_transform() and not self.tried_transform_lookup:
                self.node.get_logger().warn("Transform not available yet, skipping this point cloud")
                return
                
            # Extract point cloud data correctly - explicit XYZ extraction
            pc_points = []
            for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                pc_points.append([point[0], point[1], point[2]])
            
            points = np.array(pc_points, dtype=np.float64)
            
            if len(points) == 0:
                self.node.get_logger().warn("Received empty point cloud")
                return
            
            # Convert to Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Transform point cloud to robot base frame if we have the transform
            if self.transform_matrix is not None:
                pcd = pcd.transform(self.transform_matrix)
                self.node.get_logger().info(f"Transformed point cloud to {self.robot_base_frame} frame")
            else:
                self.node.get_logger().warn("Using point cloud in original frame - transform not available")
            
            # Create a box for cropping
            min_bound = [
                self.crop_bounds["x_min"],
                self.crop_bounds["y_min"],
                self.crop_bounds["z_min"]
            ]
            max_bound = [
                self.crop_bounds["x_max"],
                self.crop_bounds["y_max"],
                self.crop_bounds["z_max"]
            ]
            
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
            
            # Crop the point cloud
            cropped_pcd = pcd.crop(bbox)
            
            # Extract points from cropped point cloud
            cropped_points = np.asarray(cropped_pcd.points)
            
            if len(cropped_points) == 0:
                self.node.get_logger().warn("No points remain after cropping")
                return
            
            # Create top-down occupancy grid from cropped points
            grid_data = self.create_top_down_occupancy(cropped_points)
            
            if grid_data is None:
                self.node.get_logger().error("Failed to create occupancy grid")
                return
            
            # Publish the full cropped point cloud (before segmentation)
            header = msg.header
            header.frame_id = self.robot_base_frame
            
            # Create point cloud message
            cropped_points = np.array(cropped_points)
            cropped_cloud_msg = pc2.create_cloud_xyz32(header, cropped_points)
            self.cloud_publisher.publish(cropped_cloud_msg)
            self.node.get_logger().info(f"Published cropped point cloud with {len(cropped_points)} points")
            
            # Run YOLO inference directly on the grid image
            if self.model is not None:
                yolo_result = self.run_yolo_inference(grid_data["image"])
                
                # Always create debug visualization showing grid and detection boxes
                debug_markers = create_detection_debug_markers(
                    grid_data, 
                    yolo_result, 
                    self.robot_base_frame, 
                    self.grid_resolution, 
                    self.crop_bounds["z_min"],
                    logger=self.node.get_logger()
                )
                # Set timestamp
                for marker in debug_markers.markers:
                    marker.header.stamp = self.node.get_clock().now().to_msg()
                    
                self.marker_publisher.publish(debug_markers)
                self.node.get_logger().info(f"Published visualization to {self.occupancy_topic}")
                
                if yolo_result is not None and len(yolo_result.boxes) > 0:
                    # Define colors for each class (RGB format)
                    class_colors = {
                        0: (0.9, 0.2, 0.2),  # Torso - red
                        1: (0.2, 0.9, 0.2),  # Head - green
                        6: (0.7, 0.5, 0.3)   # legs - brown
                    }
                    
                    # Filter boxes to get the highest confidence detection for each class
                    boxes_by_class = {}
                    for box in yolo_result.boxes:
                        cls = int(box.cls.item())
                        conf = float(box.conf.item())
                        
                        # Only consider head (1), torso (0), and legs (6)
                        if cls not in [0, 1, 6]:
                            continue
                        
                        # Keep the highest confidence detection for each class
                        if cls not in boxes_by_class or conf > boxes_by_class[cls][1]:
                            boxes_by_class[cls] = (box, conf)
                    
                    # Extract points for each detection
                    points_by_class = {}
                    for cls, (box, conf) in boxes_by_class.items():
                        detection_points = self.get_points_in_detection(
                            cropped_points, 
                            grid_data["mask"], 
                            grid_data["points"], 
                            grid_data["origin"], 
                            box
                        )
                        
                        if detection_points is not None and len(detection_points) > 0:
                            points_by_class[cls] = detection_points
                    
                    # Create markers for detected regions
                    if points_by_class:
                        detection_markers = create_detection_markers(
                            points_by_class, 
                            class_colors, 
                            self.robot_base_frame,
                            logger=self.node.get_logger()
                        )
                        # Set timestamp
                        for marker in detection_markers.markers:
                            marker.header.stamp = self.node.get_clock().now().to_msg()
                            
                        if detection_markers.markers:
                            self.detection_publisher.publish(detection_markers)
                            self.node.get_logger().info(f"Published detection markers with {len(detection_markers.markers)} regions")
                        else:
                            self.node.get_logger().warn("No detection markers created")
                    else:
                        self.node.get_logger().warn("No points found in detection regions")
            
            # Increment frame counter
            self.frame_counter += 1
            
        except Exception as e:
            self.node.get_logger().error(f"Error processing point cloud: {str(e)}")
            import traceback
            self.node.get_logger().error(traceback.format_exc())

    def get_points_in_detection(self, points, mask, representative_points, origin, detection_box):
        """Extract points that fall within a detection bounding box"""
        try:
            # Get bounding box coordinates from YOLO result (in YOLO image coordinates)
            xyxy = detection_box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            
            # Get grid dimensions
            grid_width, grid_height = mask.shape
            
            # Get the original grid image dimensions - these are what we fed to YOLO
            # The grid image dimensions match our grid dimensions but with height/width swapped
            # because we created the image with shape (grid_height, grid_width)
            original_img_height, original_img_width = grid_height, grid_width
            
            # Get the YOLO processed image dimensions from the detection box
            # These are the dimensions YOLO resized our input to for processing
            if hasattr(detection_box, 'orig_shape'):
                yolo_height, yolo_width = detection_box.orig_shape
                self.node.get_logger().info(f"Using YOLO dimensions from result: {yolo_width}x{yolo_height}")
            else:
                # Fallback to default YOLO dimensions if not available
                yolo_width, yolo_height = 640, 640  # Default YOLO v8 dimensions
                self.node.get_logger().info(f"Using default YOLO dimensions: {yolo_width}x{yolo_height}")
            
            # Get the original coordinates from the detection box (in YOLO image coordinates)
            x1_yolo, y1_yolo, x2_yolo, y2_yolo = xyxy
            
            # Log the original YOLO coordinates
            self.node.get_logger().info(f"Original YOLO coordinates: [{x1_yolo:.1f}, {y1_yolo:.1f}, {x2_yolo:.1f}, {y2_yolo:.1f}]")
            
            # Calculate scaling factors between YOLO dimensions and original image dimensions
            x_scale = original_img_width / yolo_width
            y_scale = original_img_height / yolo_height
            
            # Scale YOLO coordinates back to original image coordinates
            x1_img = int(x1_yolo * x_scale)
            x2_img = int(x2_yolo * x_scale)
            y1_img = int(y1_yolo * y_scale)
            y2_img = int(y2_yolo * y_scale)
            
            self.node.get_logger().info(f"Scaled to original image: [{x1_img}, {y1_img}, {x2_img}, {y2_img}]")
            
            # Convert image coordinates to grid indices
            # For x-axis: Direct mapping (image x → grid x)
            x1_idx = x1_img
            x2_idx = x2_img
            
            # For y-axis: Flip the y-axis (image has origin at top-left, grid at bottom-left)
            # This means y1_img (top of box) maps to grid_height - y1_img (from bottom)
            y1_idx = grid_height - y2_img  # Note the swap of y2 and y1 due to flipping
            y2_idx = grid_height - y1_img
            
            # Ensure indices are within bounds
            x1_idx = max(0, min(grid_width - 1, x1_idx))
            x2_idx = max(0, min(grid_width - 1, x2_idx))
            y1_idx = max(0, min(grid_height - 1, y1_idx))
            y2_idx = max(0, min(grid_height - 1, y2_idx))
            
            # Make sure x1 < x2 and y1 < y2
            x1_idx, x2_idx = min(x1_idx, x2_idx), max(x1_idx, x2_idx)
            y1_idx, y2_idx = min(y1_idx, y2_idx), max(y1_idx, y2_idx)
            
            # Log the converted grid indices
            self.node.get_logger().info(f"Detection box: YOLO={xyxy}, Grid=[{x1_idx},{y1_idx},{x2_idx},{y2_idx}]")
            
            # Calculate real-world coordinates for debugging
            x_origin, y_origin = origin
            x1_world = x_origin + x1_idx * self.grid_resolution
            y1_world = y_origin + y1_idx * self.grid_resolution
            x2_world = x_origin + (x2_idx + 1) * self.grid_resolution
            y2_world = y_origin + (y2_idx + 1) * self.grid_resolution
            self.node.get_logger().info(f"World coordinates: [{x1_world:.2f}, {y1_world:.2f}, {x2_world:.2f}, {y2_world:.2f}]")
            
            # Create a mask for this detection
            detection_mask = np.zeros_like(mask)
            detection_mask[x1_idx:x2_idx+1, y1_idx:y2_idx+1] = mask[x1_idx:x2_idx+1, y1_idx:y2_idx+1]
            
            # Get representative points for this detection
            detection_points = []
            for i in range(x1_idx, x2_idx+1):
                for j in range(y1_idx, y2_idx+1):
                    if detection_mask[i, j] and representative_points[i, j] is not None:
                        detection_points.append(representative_points[i, j])
            
            self.node.get_logger().info(f"Found {len(detection_points)} points in detection region")
            return np.array(detection_points) if detection_points else None
            
        except Exception as e:
            self.node.get_logger().error(f"Error extracting points for detection: {str(e)}")
            import traceback
            self.node.get_logger().error(traceback.format_exc())
            return None

def cli() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Transform and crop point cloud data for massage planning"
    )
    parser.add_argument(
        "--input-topic", type=str, default="/depth/color/points",
        help="Input PointCloud2 topic to subscribe to"
    )
    parser.add_argument(
        "--output-topic", type=str, default="/massage_planning/cropped_cloud",
        help="Output topic for the transformed and cropped point cloud"
    )
    parser.add_argument(
        "--occupancy-topic", type=str, default="/massage_planning/debug_visualization",
        help="Output topic for visualization"
    )
    parser.add_argument(
        "--camera-frame", type=str, default="camera_depth_optical_frame",
        help="Camera frame ID"
    )
    parser.add_argument(
        "--robot-base-frame", type=str, default="j2n6s300_link_base",
        help="Robot base frame ID"
    )
    parser.add_argument(
        "--tf-prefix", type=str, default=None,
        help="TF prefix for frame names"
    )
    parser.add_argument(
        "--x-min", type=float, default=-0.7,
        help="Minimum X value (in robot base frame) for cropping"
    )
    parser.add_argument(
        "--x-max", type=float, default=1.4,
        help="Maximum X value (in robot base frame) for cropping"
    )
    parser.add_argument(
        "--y-min", type=float, default=-0.8,
        help="Minimum Y value (in robot base frame) for cropping"
    )
    parser.add_argument(
        "--y-max", type=float, default=-0.1,
        help="Maximum Y value (in robot base frame) for cropping"
    )
    parser.add_argument(
        "--z-min", type=float, default=.05,
        help="Minimum Z value (in robot base frame) for cropping"
    )
    parser.add_argument(
        "--z-max", type=float, default=.5,
        help="Maximum Z value (in robot base frame) for cropping"
    )
    parser.add_argument(
        "--grid-resolution", type=float, default=0.01,
        help="Resolution of the occupancy grid in meters (default: 1cm)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="~/pointcloud_plys",
        help="Directory to save PLY files (if save-plys is set)"
    )
    parser.add_argument(
        "--save-plys", action="store_true",
        help="Save PLY files of processed point clouds"
    )
    parser.add_argument(
        "--save-grids", action="store_true",
        help="Save occupancy grid as PNG images"
    )
    return parser


@ros_process.main(cli(), uses_tf=True)
def main(args: argparse.Namespace) -> None:
    """Main entry point for the ROS node"""
    PointCloudTransformerAndOccupancyMapper(
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        occupancy_topic=args.occupancy_topic,
        camera_frame=args.camera_frame,
        robot_base_frame=args.robot_base_frame,
        tf_prefix=args.tf_prefix,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        z_min=args.z_min,
        z_max=args.z_max,
        grid_resolution=args.grid_resolution,
        output_dir=args.output_dir,
        save_plys=args.save_plys,
        save_grids=args.save_grids
    )
    wait_for_shutdown()


if __name__ == "__main__":
    main()