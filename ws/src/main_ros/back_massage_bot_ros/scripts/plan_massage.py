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
import rclpy

import synchros2.process as ros_process
import synchros2.scope as ros_scope
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped, PoseStamped
from synchros2.context import wait_for_shutdown
from synchros2.utilities import namespace_with
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String
from cv_bridge import CvBridge
from std_srvs.srv import Trigger, Empty
import tf2_ros
from tf2_geometry_msgs import PoseStamped as TF2PoseStamped

# Import only the utilities we actually need
from utils import (
    create_yolo_markers,
    create_top_down_occupancy,
    get_best_detections,
    run_yolo_inference,
    publish_image,
    create_detailed_back_regions,
    get_points_in_detection,
    create_massage_region_markers,
    create_numbered_point_markers,
    create_massage_motion_plan
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
        model_path: str = "/back_massage_bot/src/back_massage_bot/models/third_runs_the_charm/weights/best.pt",
        yolo_default_size: tuple = (640, 640),  # Default YOLO input size (width, height)
        yolo_low_conf_threshold: float = 0.1,  # Lower confidence threshold for fallback detection
        spine_width_fraction: float = 6.0,  # Spine width as fraction of torso width (1/6)
        back_regions_count: int = 3,  # Number of vertical regions to divide the back into
        point_stride: int = 5,  # Stride for point numbering visualization
        massage_gun_tip_transform: list = [0.0, 0.0, 0.05],  # Default 5cm upward offset
        visualize_path: bool = True  # Whether to visualize the massage gun tip path
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
        self.yolo_default_size = yolo_default_size
        self.yolo_low_conf_threshold = yolo_low_conf_threshold
        self.spine_width_fraction = spine_width_fraction
        self.back_regions_count = back_regions_count
        self.point_stride = point_stride
        self.massage_gun_tip_transform = massage_gun_tip_transform
        self.visualize_path = visualize_path
        
        # Define topic names for visualization and debugging
        self.regions_topic = "/massage_planning/body_regions"
        self.yolo_input_topic = "/massage_planning/yolo_input"
        self.yolo_output_topic = "/massage_planning/yolo_output"
        self.grid_image_topic = "/massage_planning/grid_image"
        
        # Define colors for body regions
        self.class_colors = {
            0: (0.9, 0.2, 0.2),  # Torso - red
            1: (0.2, 0.9, 0.2),  # Head - green
            6: (0.7, 0.5, 0.3)   # legs - brown
        }
        
        # Create a unique directory for this run if needed
        if self.save_plys or self.save_grids:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(self.output_dir, f"run_{timestamp}")
            os.makedirs(self.run_dir, exist_ok=True)
            self.node.get_logger().debug(f"Created output directory at {self.run_dir}")
            
            if self.save_grids:
                self.grid_dir = os.path.join(self.run_dir, "grids")
                os.makedirs(self.grid_dir, exist_ok=True)
                self.node.get_logger().debug(f"Will save grid images to {self.grid_dir}")
        else:
            self.run_dir = None
            self.grid_dir = None
        
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
        self.yolo_input_pub = self.node.create_publisher(Image, self.yolo_input_topic, qos)
        self.yolo_output_pub = self.node.create_publisher(Image, self.yolo_output_topic, qos)
        self.grid_image_pub = self.node.create_publisher(Image, self.grid_image_topic, qos)
        
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
        self.node.get_logger().debug("Started point cloud processing thread")
        
        # Subscribe to the point cloud topic
        self.subscription = self.node.create_subscription(
            PointCloud2,
            self.input_topic,
            self.point_cloud_callback,
            qos
        )
        
        self.node.get_logger().debug(f"Subscribed to {self.input_topic}")
        self.node.get_logger().debug(f"Publishing transformed and cropped point cloud to {self.output_topic}")
        self.node.get_logger().debug(f"Publishing visualization to {self.occupancy_topic}")
        self.node.get_logger().info("Waiting for point clouds...")
        
        # Initialize YOLO model for inference
        self.node.get_logger().info(f"Attempting to load YOLO model from {model_path}")
        if not os.path.exists(model_path):
            self.node.get_logger().error(f"Model file does not exist: {model_path}")
        
        try:
            self.model = ultralytics.YOLO(model_path)
            self.model.args['data'] = "data/data.yaml"
            # Disable verbose output from YOLO
            self.model.args['verbose'] = False
            self.node.get_logger().info(f"Successfully loaded YOLO model from {model_path}")
        except Exception as e:
            self.model = None
            self.node.get_logger().error(f"Failed to load YOLO model: {str(e)}")
            import traceback
            self.node.get_logger().error(traceback.format_exc())
        
        # Create additional publisher for detected regions
        self.detection_publisher = self.node.create_publisher(MarkerArray, self.regions_topic, qos)
        self.node.get_logger().debug(f"Will publish detected regions to {self.regions_topic}")
        
        # Add these to the __init__ method
        self.plan_and_execute_service = self.node.create_service(
            Trigger, 
            '/plan_and_execute_massage', 
            self.plan_and_execute_callback
        )
        
        # Create services for each of the 6 back regions
        self.region_services = {}
        for region in ["left_upper", "left_middle", "left_lower", 
                      "right_upper", "right_middle", "right_lower"]:
            service_name = f'/massage_region/{region}'
            self.region_services[region] = self.node.create_service(
                Empty,
                service_name,
                lambda req, resp, region=region: self.region_service_callback(req, resp, region)
            )
            self.node.get_logger().debug(f"Created service for region: {service_name}")
        
        self.region_selection_sub = self.node.create_subscription(
            String,
            '/massage_region_selection',
            self.region_selection_callback,
            10
        )
        
        self.arm_dispatch_pub = self.node.create_publisher(
            PoseStamped,
            '/arm_dispatch_command',
            10
        )
        
        self.return_home_client = self.node.create_client(
            Trigger,
            '/return_to_home'
        )
        
        self.selected_region = None
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self.node)
        
        # Default to left_upper if no region is selected
        self.selected_region = "left_upper"

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
            self.node.get_logger().debug(f"Successfully cached transform from {self.camera_frame} to {self.robot_base_frame}")
            return True
            
        except Exception as e:
            self.node.get_logger().error(f"Failed to get transform: {str(e)}")
            return False

    def point_cloud_callback(self, msg: PointCloud2) -> None:
        """Callback for point cloud messages - just adds to queue"""
        try:
            # Add the message to the queue, with a timeout to avoid blocking
            self.point_cloud_queue.put(msg, block=True, timeout=0.1)
            self.node.get_logger().debug(f"Added point cloud to queue (size: {self.point_cloud_queue.qsize()})")
        except queue.Full:
            # Only log queue full warning occasionally to reduce verbosity
            if hasattr(self, 'last_queue_warning_time'):
                current_time = time.time()
                if current_time - self.last_queue_warning_time > 5.0:  # Only warn every 5 seconds
                    self.node.get_logger().warn("Point cloud queue is full, dropping messages")
                    self.last_queue_warning_time = current_time
            else:
                self.last_queue_warning_time = time.time()
                self.node.get_logger().warn("Point cloud queue is full, dropping message")
        except Exception as e:
            self.node.get_logger().error(f"Error in point cloud callback: {str(e)}")

    def process_point_clouds(self):
        """Process point clouds from the queue in a separate thread"""
        self.node.get_logger().debug("Point cloud processing thread started")
        
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
                self.node.get_logger().debug(f"Transformed point cloud to {self.robot_base_frame} frame")
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
            
            # Create top-down occupancy grid from cropped points using the utility function
            grid_data = create_top_down_occupancy(
                cropped_points, 
                self.crop_bounds, 
                self.grid_resolution,
                save_grids=self.save_grids,
                grid_dir=self.grid_dir,
                frame_counter=self.frame_counter,
                logger=self.node.get_logger()
            )
            
            if grid_data is None:
                self.node.get_logger().error("Failed to create occupancy grid")
                return
            
            # Publish the grid image used for YOLO detection
            current_time = self.node.get_clock().now().to_msg()
            publish_image(
                grid_data["image"], 
                self.grid_image_pub, 
                self.robot_base_frame, 
                self.cv_bridge,
                timestamp=current_time,
                log_message="Published original grid image used for YOLO detection",
                logger=self.node.get_logger()
            )
            
            # Publish the full cropped point cloud (before segmentation)
            header = msg.header
            header.frame_id = self.robot_base_frame
            
            # Create point cloud message
            cropped_points = np.array(cropped_points)
            cropped_cloud_msg = pc2.create_cloud_xyz32(header, cropped_points)
            self.cloud_publisher.publish(cropped_cloud_msg)
            self.node.get_logger().debug(f"Published cropped point cloud with {len(cropped_points)} points")
            
            # Run YOLO inference directly on the grid image using the utility function
            if self.model is not None:
                yolo_result = run_yolo_inference(
                    self.model, 
                    grid_data["image"],
                    yolo_low_conf_threshold=self.yolo_low_conf_threshold,
                    cv_bridge=self.cv_bridge,
                    yolo_input_pub=self.yolo_input_pub,
                    yolo_output_pub=self.yolo_output_pub,
                    robot_base_frame=self.robot_base_frame,
                    logger=self.node.get_logger()
                )
                
                if yolo_result is not None:
                    # Get the best detection for each class using the utility function
                    best_detections = get_best_detections(yolo_result, logger=self.node.get_logger())
                    
                    # STEP 1: Identify regions - separate this from planning
                    points_by_class, detailed_regions = self.identify_massage_regions(
                        best_detections,
                        cropped_points,
                        grid_data,
                        self.class_colors,
                        self.robot_base_frame,
                        detection_publisher=self.detection_publisher,
                        marker_publisher=self.marker_publisher,
                        crop_bounds=self.crop_bounds,
                        logger=self.node.get_logger()
                    )
                    
                    # STEP 2: Create motion plans for each region
                    region_motion_plans = self.create_region_motion_plans(
                        detailed_regions,
                        point_stride=self.point_stride,
                        massage_gun_tip_transform=self.massage_gun_tip_transform,
                        visualize_path=self.visualize_path
                    )
                    
                    # Store the results for later use
                    self.latest_points_by_class = points_by_class
                    self.latest_detailed_regions = detailed_regions
                    self.latest_region_motion_plans = region_motion_plans
                
                else:
                    # Create empty debug visualization if no detections
                    debug_markers = create_yolo_markers(
                        grid_data, 
                        None,  # No detections
                        self.robot_base_frame, 
                        self.grid_resolution, 
                        self.crop_bounds["z_min"],
                        logger=self.node.get_logger()
                    )
                    self.marker_publisher.publish(debug_markers)
            
            # Increment frame counter
            self.frame_counter += 1
            
        except Exception as e:
            self.node.get_logger().error(f"Error processing point cloud: {str(e)}")
            import traceback
            self.node.get_logger().error(traceback.format_exc())

    def identify_massage_regions(self, best_detections, cropped_points, grid_data, class_colors, 
                                robot_base_frame, detection_publisher=None, marker_publisher=None, 
                                crop_bounds=None, logger=None):
        """
        Identify massage regions from YOLO detections
        
        Args:
            best_detections: Dictionary of best detections for each class
            cropped_points: Array of 3D points from the cropped point cloud
            grid_data: Dictionary with grid information
            class_colors: Dictionary mapping class IDs to RGB color tuples
            robot_base_frame: Frame ID for the markers
            detection_publisher: Publisher for detection markers
            marker_publisher: Publisher for debug markers
            crop_bounds: Dictionary with crop bounds
            logger: Optional ROS logger for debug messages
            
        Returns:
            Tuple of (points_by_class, detailed_regions)
        """
        # Extract points for each detection
        points_by_class = {}
        
        for class_instance_id, (box, conf) in best_detections.items():
            detection_points = get_points_in_detection(
                cropped_points, 
                grid_data, 
                box,
                structured_pattern=True,  # Always use structured pattern
                logger=logger
            )
            
            if detection_points is not None and len(detection_points) > 0:
                points_by_class[class_instance_id] = detection_points
        
        # Check if we have both torso and legs to create detailed back regions
        detailed_regions = {}
        
        if 0 in points_by_class and 6 in points_by_class:
            # Create detailed back regions
            detailed_regions = create_detailed_back_regions(
                points_by_class[0],  # torso points
                points_by_class[6],  # legs points
                spine_width_fraction=self.spine_width_fraction,
                back_regions_count=self.back_regions_count,
                logger=logger
            )
        
        # Create markers for detected regions
        if points_by_class or detailed_regions:
            detection_markers = create_massage_region_markers(
                points_by_class, 
                class_colors, 
                robot_base_frame,
                detailed_regions=detailed_regions,
                logger=logger
            )
            
            # Publish detection markers if publisher is provided
            if detection_publisher:
                detection_publisher.publish(detection_markers)
                if logger:
                    logger.debug("Published detection markers")
        
        return points_by_class, detailed_regions

    def apply_massage_offset(self, massage_point_matrix):
        """
        Transform from tool tip position (on body) to end effector position.
        First applies rotation to get the correct frame, then applies the full XYZ translation.
        
        Args:
            massage_point_matrix: 4x4 homogeneous matrix for the tool tip position (on body)
            
        Returns:
            PoseStamped message representing the end effector pose in global frame
        """
        try:
            # Get the tool tip position (on body)
            tool_tip_point = massage_point_matrix[:3, 3]
            
            # Determine if this is a left region based on Y coordinate
            is_left_region = tool_tip_point[1] < -0.3  # Threshold for left vs right
            
            # Create the tool tip frame with appropriate orientation
            tool_tip_frame = np.eye(4)
            if is_left_region:
                # For left regions, tilt around X to extend reach
                R_tilt = Rotation.from_euler('X', -30, degrees=True)
            else:
                # For right regions, keep straight up
                R_tilt = Rotation.from_euler('X', 0, degrees=True)
            
            # Apply the rotation to the tool tip frame
            tool_tip_frame[:3, :3] = R_tilt.as_matrix()
            
            # Create the offset transform from tool tip to end effector
            # This moves the end effector back along all axes in the rotated frame
            offset_transform = np.eye(4)
            offset_transform[0, 3] = -self.massage_gun_tip_transform[0]  # Move back along X
            offset_transform[1, 3] = -self.massage_gun_tip_transform[1]  # Move along Y
            offset_transform[2, 3] = -self.massage_gun_tip_transform[2]  # Move along Z
            
            # Combine the transforms: tool_tip -> orientation -> offset
            # This gives us the end effector position and orientation
            end_effector_matrix = massage_point_matrix @ tool_tip_frame @ offset_transform
            
            # Create pose message from the final transform
            pose = PoseStamped()
            pose.header.stamp = self.node.get_clock().now().to_msg()
            pose.header.frame_id = self.robot_base_frame
            
            # Set position from the final transform
            pose.pose.position.x = end_effector_matrix[0, 3]
            pose.pose.position.y = end_effector_matrix[1, 3]
            pose.pose.position.z = end_effector_matrix[2, 3]
            
            # Set orientation from the final transform
            R_final = Rotation.from_matrix(end_effector_matrix[:3, :3])
            quat = R_final.as_quat()
            pose.pose.orientation.w = quat[3]
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            
            return pose
            
        except Exception as e:
            self.node.get_logger().error(f"Error applying massage offset: {str(e)}")
            import traceback
            self.node.get_logger().error(traceback.format_exc())
            # Return a default pose if something fails
            default_pose = PoseStamped()
            default_pose.header.stamp = self.node.get_clock().now().to_msg()
            default_pose.header.frame_id = self.robot_base_frame
            default_pose.pose.position.x = massage_point_matrix[0, 3]
            default_pose.pose.position.y = massage_point_matrix[1, 3]
            default_pose.pose.position.z = massage_point_matrix[2, 3]
            quat = Rotation.from_matrix(massage_point_matrix[:3, :3]).as_quat()
            default_pose.pose.orientation.w = quat[3]
            default_pose.pose.orientation.x = quat[0]
            default_pose.pose.orientation.y = quat[1]
            default_pose.pose.orientation.z = quat[2]
            return default_pose

    def create_region_motion_plans(self, detailed_regions, point_stride=5, 
                                  massage_gun_tip_transform=None, visualize_path=True):
        """
        Create motion plans for each detected back region
        
        Args:
            detailed_regions: Dictionary of detailed back regions
            point_stride: Stride for point selection in motion planning
            massage_gun_tip_transform: Transform from end effector to tool tip location [x,y,z]
            visualize_path: Whether to visualize the massage gun tip path
            
        Returns:
            Dictionary mapping region names to motion plans with position and orientation
        """
        region_motion_plans = {}
        logger = self.node.get_logger()
        
        # Process each region
        for region_name, (region_points, region_color) in detailed_regions.items():
            # Skip the spine region for motion planning
            if region_name == "spine":
                logger.debug(f"Skipping motion planning for spine region")
                continue
            
            # Create motion plan for this region
            motion_plan = create_massage_motion_plan(
                region_points,
                stride=point_stride,
                massage_gun_tip_transform=None,  # No transform needed here - keep in base frame
                logger=logger
            )
            
            # Add orientation to all points in the motion plan
            motion_plan_with_orientation = {
                'rows': [],
                'connections': [],
                'all_points': []
            }
            
            # Create transforms for all points in rows
            for row_idx, row in enumerate(motion_plan['rows']):
                row_with_orientation = []
                for point_idx, point in enumerate(row):
                    # Calculate the overall point index for this point
                    overall_idx = row_idx * len(row) + point_idx
                    
                    # Create a transform for the massage point (Z down orientation)
                    massage_transform = TransformStamped()
                    massage_transform.header.stamp = self.node.get_clock().now().to_msg()
                    massage_transform.header.frame_id = self.robot_base_frame
                    massage_transform.child_frame_id = f"massage_point_{region_name}_{overall_idx}"
                    
                    # Set position
                    massage_transform.transform.translation.x = point[0]
                    massage_transform.transform.translation.y = point[1]
                    massage_transform.transform.translation.z = point[2]
                    
                    # Set Z down orientation (180° about X) using scipy Rotation
                    R = Rotation.from_euler('X', 180, degrees=True)
                    quat = R.as_quat()  # Returns [x, y, z, w]
                    massage_transform.transform.rotation.w = quat[3]
                    massage_transform.transform.rotation.x = quat[0]
                    massage_transform.transform.rotation.y = quat[1]
                    massage_transform.transform.rotation.z = quat[2]
                    
                    # Store the transform for later use
                    row_with_orientation.append(massage_transform)
                motion_plan_with_orientation['rows'].append(row_with_orientation)
            
            # Add orientation to connections with intermediate higher points
            for start_idx, (start_point, end_point) in enumerate(motion_plan['connections']):
                # Create an intermediate point that's higher than both start and end points
                intermediate_z_offset = 0.03  # 3cm higher
                max_z = max(start_point[2], end_point[2])
                intermediate_point = [
                    (start_point[0] + end_point[0]) / 2.0,
                    (start_point[1] + end_point[1]) / 2.0,
                    max_z + intermediate_z_offset
                ]
                
                # Calculate indices for the points
                start_overall_idx = start_idx * 2
                intermediate_idx = start_overall_idx + 1
                end_idx = start_overall_idx + 2
                
                # Create transforms for start, intermediate, and end points
                for idx, point in [(start_overall_idx, start_point), 
                                 (intermediate_idx, intermediate_point),
                                 (end_idx, end_point)]:
                    massage_transform = TransformStamped()
                    massage_transform.header.stamp = self.node.get_clock().now().to_msg()
                    massage_transform.header.frame_id = self.robot_base_frame
                    massage_transform.child_frame_id = f"massage_point_{region_name}_{idx}"
                    
                    # Set position
                    massage_transform.transform.translation.x = point[0]
                    massage_transform.transform.translation.y = point[1]
                    massage_transform.transform.translation.z = point[2]
                    
                    # Set Z down orientation (180° about X) using scipy Rotation
                    R = Rotation.from_euler('X', 180, degrees=True)
                    quat = R.as_quat()  # Returns [x, y, z, w]
                    massage_transform.transform.rotation.w = quat[3]
                    massage_transform.transform.rotation.x = quat[0]
                    massage_transform.transform.rotation.y = quat[1]
                    massage_transform.transform.rotation.z = quat[2]
                    
                    # Store the transform
                    if idx == start_overall_idx:
                        transformed_start = massage_transform
                    elif idx == intermediate_idx:
                        transformed_intermediate = massage_transform
                    else:
                        transformed_end = massage_transform
                
                # Add the connection with the intermediate point
                motion_plan_with_orientation['connections'].append(
                    (transformed_start, transformed_intermediate)
                )
                motion_plan_with_orientation['connections'].append(
                    (transformed_intermediate, transformed_end)
                )
            
            # Rebuild the all_points list to include the intermediate points
            all_points_with_orientation = []
            
            # First, add all row points in order
            for row in motion_plan_with_orientation['rows']:
                all_points_with_orientation.extend(row)
            
            # Then add the connection points
            for i in range(0, len(motion_plan_with_orientation['connections']), 2):
                if i+1 < len(motion_plan_with_orientation['connections']):
                    # Add the intermediate point
                    intermediate_point = motion_plan_with_orientation['connections'][i][1]
                    all_points_with_orientation.append(intermediate_point)
                    
                    # Add the end point of the second connection (which is the start of the next row)
                    next_row_start = motion_plan_with_orientation['connections'][i+1][1]
                    all_points_with_orientation.append(next_row_start)
            
            motion_plan_with_orientation['all_points'] = all_points_with_orientation
            
            # Store the motion plan with the region color
            region_motion_plans[region_name] = (motion_plan_with_orientation, region_color)
            
            # Visualize the motion plan if requested
            if visualize_path:
                # Create a unique namespace for each region
                namespace = f"numbered_points_region_{region_name}"
                
                logger.debug(f"Creating markers for region {region_name} with color {region_color}")
                
                # Create numbered point markers with section-based numbering for points on the body
                numbered_markers = create_numbered_point_markers(
                    region_points,
                    self.robot_base_frame,
                    marker_namespace=namespace,
                    stride=point_stride,
                    section_based=True,  # Enable section-based numbering
                    logger=logger,
                    massage_gun_tip_transform=None,  # No transform needed here - keep in base frame
                    visualize_path=visualize_path,
                    region_color=region_color
                )
                
                # Create markers for end effector positions (off the body)
                end_effector_markers = MarkerArray()
                for i, transform in enumerate(all_points_with_orientation):
                    try:
                        # Create a marker for the end effector position
                        marker = Marker()
                        marker.header.frame_id = self.robot_base_frame
                        marker.header.stamp = self.node.get_clock().now().to_msg()
                        marker.ns = f"{namespace}_end_effector"
                        marker.id = i
                        marker.type = Marker.SPHERE
                        marker.action = Marker.ADD
                        
                        # Get the point and rotation from the transform
                        point = [
                            transform.transform.translation.x,
                            transform.transform.translation.y,
                            transform.transform.translation.z
                        ]
                        rotation = [
                            transform.transform.rotation.w,
                            transform.transform.rotation.x,
                            transform.transform.rotation.y,
                            transform.transform.rotation.z
                        ]
                        
                        # Create 4x4 matrix from transform
                        massage_point_matrix = np.eye(4)
                        massage_point_matrix[:3, 3] = point
                        massage_point_matrix[:3, :3] = Rotation.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]]).as_matrix()
                        
                        # Apply the massage offset to get the end effector position
                        end_effector_pose = self.apply_massage_offset(massage_point_matrix)
                        
                        # Set position and orientation from the end effector pose
                        marker.pose.position.x = end_effector_pose.pose.position.x
                        marker.pose.position.y = end_effector_pose.pose.position.y
                        marker.pose.position.z = end_effector_pose.pose.position.z
                        marker.pose.orientation = end_effector_pose.pose.orientation
                        
                        # Set scale (smaller than body points)
                        marker.scale.x = 0.02  # 2cm diameter
                        marker.scale.y = 0.02
                        marker.scale.z = 0.02
                        
                        # Set color (slightly transparent version of region color)
                        marker.color.r = region_color[0]
                        marker.color.g = region_color[1]
                        marker.color.b = region_color[2]
                        marker.color.a = 0.5  # 50% transparency
                        
                        # Add text label
                        marker.text = f"EE_{i}"
                        
                        end_effector_markers.markers.append(marker)
                    except Exception as e:
                        logger.error(f"Error creating marker for point {i}: {str(e)}")
                        continue
                
                # Combine both marker arrays
                all_markers = MarkerArray()
                all_markers.markers.extend(numbered_markers.markers)
                all_markers.markers.extend(end_effector_markers.markers)
                
                # Publish all markers
                self.marker_publisher.publish(all_markers)
                logger.debug(f"Published markers for region {region_name} (stride={point_stride})")
        
        return region_motion_plans

    def plan_and_execute_callback(self, request, response):
        """Callback for the plan and execute massage service"""
        try:
            self.node.get_logger().info("Received request to plan and execute massage")
            
            # Check if we have motion plans
            if not hasattr(self, 'latest_region_motion_plans') or not self.latest_region_motion_plans:
                self.node.get_logger().error("No motion plans available. Please wait for point cloud processing.")
                response.success = False
                response.message = "No motion plans available. Please wait for point cloud processing."
                return response
            
            # Check if we have a selected region
            if not self.selected_region:
                self.node.get_logger().error("No region selected. Please select a region first.")
                response.success = False
                response.message = "No region selected. Please select a region first."
                return response
            
            # Check if the selected region exists in our motion plans
            if self.selected_region not in self.latest_region_motion_plans:
                self.node.get_logger().error(f"Region {self.selected_region} not found in motion plans. Available regions: {list(self.latest_region_motion_plans.keys())}")
                response.success = False
                response.message = f"Region {self.selected_region} not found in motion plans. Available regions: {list(self.latest_region_motion_plans.keys())}"
                return response
            
            # Execute the motion plan for the selected region
            success = self.execute_motion_plan(self.selected_region)
            
            # Return home after execution
            self.call_return_home_service()
            
            # Set response
            response.success = success
            if success:
                response.message = f"Successfully executed massage for region: {self.selected_region}"
            else:
                response.message = f"Failed to execute massage for region: {self.selected_region}"
            
            return response
        except Exception as e:
            self.node.get_logger().error(f"Error in plan and execute callback: {str(e)}")
            import traceback
            self.node.get_logger().error(traceback.format_exc())
            response.success = False
            response.message = f"Error executing massage: {str(e)}"
            return response

    def region_service_callback(self, request, response, region):
        """Callback for region-specific services"""
        try:
            self.node.get_logger().info(f"Received request to massage region: {region}")
            
            # Set the selected region
            self.selected_region = region
            
            # Check if we have motion plans
            if not hasattr(self, 'latest_region_motion_plans') or not self.latest_region_motion_plans:
                self.node.get_logger().error("No motion plans available. Please wait for point cloud processing.")
                return response
            
            # Check if the region exists in our motion plans
            if region not in self.latest_region_motion_plans:
                self.node.get_logger().error(f"Region {region} not found in motion plans. Available regions: {list(self.latest_region_motion_plans.keys())}")
                return response
            
            # Get the motion plan for the selected region
            motion_plan, region_color = self.latest_region_motion_plans[region]
            
            # Extract the points with orientation
            points_with_orientation = motion_plan['all_points']
            
            if not points_with_orientation:
                self.node.get_logger().warn(f"No points in motion plan for region: {region}")
                return response
            
            # Create markers and publish TF frames for end effector positions
            end_effector_markers = MarkerArray()
            namespace = f"end_effector_positions_{region}"
            
            for i, transform in enumerate(points_with_orientation):
                try:
                    # Get the point and rotation from the transform
                    point = [
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z
                    ]
                    rotation = [
                        transform.transform.rotation.w,
                        transform.transform.rotation.x,
                        transform.transform.rotation.y,
                        transform.transform.rotation.z
                    ]
                    
                    # Create 4x4 matrix from transform
                    massage_point_matrix = np.eye(4)
                    massage_point_matrix[:3, 3] = point
                    massage_point_matrix[:3, :3] = Rotation.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]]).as_matrix()
                    
                    # Apply the massage offset to get the end effector position
                    end_effector_pose = self.apply_massage_offset(massage_point_matrix)
                    
                    # Create a transform for the end effector position
                    end_effector_transform = TransformStamped()
                    end_effector_transform.header.stamp = self.node.get_clock().now().to_msg()
                    end_effector_transform.header.frame_id = self.robot_base_frame
                    end_effector_transform.child_frame_id = f"end_effector_{region}_{i}"
                    
                    # Set position and orientation from the end effector pose
                    end_effector_transform.transform.translation.x = end_effector_pose.pose.position.x
                    end_effector_transform.transform.translation.y = end_effector_pose.pose.position.y
                    end_effector_transform.transform.translation.z = end_effector_pose.pose.position.z
                    end_effector_transform.transform.rotation = end_effector_pose.pose.orientation
                    
                    # Publish the end effector transform
                    self.tf_broadcaster.sendTransform(end_effector_transform)
                    
                    # Create a marker for visualization
                    marker = Marker()
                    marker.header.frame_id = self.robot_base_frame
                    marker.header.stamp = self.node.get_clock().now().to_msg()
                    marker.ns = namespace
                    marker.id = i
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    
                    # Set position and orientation from the end effector pose
                    marker.pose.position.x = end_effector_pose.pose.position.x
                    marker.pose.position.y = end_effector_pose.pose.position.y
                    marker.pose.position.z = end_effector_pose.pose.position.z
                    marker.pose.orientation = end_effector_pose.pose.orientation
                    
                    # Set scale (smaller than body points)
                    marker.scale.x = 0.02  # 2cm diameter
                    marker.scale.y = 0.02
                    marker.scale.z = 0.02
                    
                    # Set color (slightly transparent version of region color)
                    marker.color.r = region_color[0]
                    marker.color.g = region_color[1]
                    marker.color.b = region_color[2]
                    marker.color.a = 0.5  # 50% transparency
                    
                    # Add text label
                    marker.text = f"EE_{i}"
                    
                    end_effector_markers.markers.append(marker)
                except Exception as e:
                    self.node.get_logger().error(f"Error creating marker for point {i}: {str(e)}")
                    continue
            
            # Publish the end effector markers
            self.marker_publisher.publish(end_effector_markers)
            self.node.get_logger().debug(f"Published end effector markers for region {region}")
            
            # Wait a moment for transforms to be published
            time.sleep(0.5)
            
            # Execute the motion plan for the selected region
            success = self.execute_motion_plan(region)
            
            # Return home after execution
            self.call_return_home_service()
            
            return response
        except Exception as e:
            self.node.get_logger().error(f"Error in region service callback: {str(e)}")
            import traceback
            self.node.get_logger().error(traceback.format_exc())
            return response

    def call_return_home_service(self):
        """Call the return to home service"""
        self.node.get_logger().info("Calling return to home service")
        
        try:
            # Check if service is available
            if not self.return_home_client.service_is_ready():
                self.node.get_logger().error("Return to home service is not available")
                return False
                
            # Create a request
            request = Trigger.Request()
            
            # Call the service (non-blocking)
            future = self.return_home_client.call_async(request)
            
            # We can't use spin_until_future_complete in synchros2 framework
            # Instead, we'll just log that we've sent the request
            self.node.get_logger().info("Sent return to home request (async)")
            
            # Return true to indicate we've sent the request
            return True
                
        except Exception as e:
            self.node.get_logger().error(f"Error calling return to home service: {str(e)}")
            import traceback
            self.node.get_logger().error(traceback.format_exc())
            return False

    def execute_motion_plan(self, region_name):
        """Execute the motion plan for a specific region"""
        try:
            # Get the motion plan for the selected region
            motion_plan, region_color = self.latest_region_motion_plans[region_name]
            
            # Extract the points with orientation
            points_with_orientation = motion_plan['all_points']
            
            if not points_with_orientation:
                self.node.get_logger().warn(f"No points in motion plan for region: {region_name}")
                return False
            
            self.node.get_logger().info(f"Executing motion plan for {region_name} with {len(points_with_orientation)} points")
            
            # Create end effector poses with tool offset for each point
            end_effector_poses = []
            for i, transform in enumerate(points_with_orientation):
                try:
                    # Get the point and rotation from the transform
                    point = [
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z
                    ]
                    rotation = [
                        transform.transform.rotation.w,
                        transform.transform.rotation.x,
                        transform.transform.rotation.y,
                        transform.transform.rotation.z
                    ]
                    
                    # Create 4x4 matrix from transform
                    massage_point_matrix = np.eye(4)
                    massage_point_matrix[:3, 3] = point
                    massage_point_matrix[:3, :3] = Rotation.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]]).as_matrix()
                    
                    # Apply the massage offset to get the end effector position
                    end_effector_pose = self.apply_massage_offset(massage_point_matrix)
                    end_effector_poses.append(end_effector_pose)
                    
                except Exception as e:
                    self.node.get_logger().error(f"Error creating end effector pose for point {i}: {str(e)}")
                    continue
            
            # Add a 3-minute delay with cancellation capability
            self.node.get_logger().info("Waiting 3 minutes before executing motion plan. Press Ctrl+C to cancel.")
            try:
                # Wait for 3 minutes (180 seconds)
                for i in range(3):
                    time.sleep(1)
                    if i % 10 == 0:  # Log every 10 seconds
                        self.node.get_logger().info(f"Waiting... {180-i} seconds remaining")
            except KeyboardInterrupt:
                self.node.get_logger().warn("Motion plan execution cancelled by user")
                return False
            
            # Execute each point in the motion plan using the end effector poses
            for i, end_effector_pose in enumerate(end_effector_poses):
                # Publish the pose command
                self.arm_dispatch_pub.publish(end_effector_pose)
                self.node.get_logger().info(f"Moving to point {i+1}/{len(end_effector_poses)} in {region_name}")
                
                # Wait for the arm to reach the position
                time.sleep(2.0)  # Adjust this delay based on your robot's speed
            
            return True
            
        except Exception as e:
            self.node.get_logger().error(f"Error executing motion plan: {str(e)}")
            import traceback
            self.node.get_logger().error(traceback.format_exc())
            return False

    def region_selection_callback(self, msg):
        """Callback for region selection messages"""
        self.selected_region = msg.data
        self.node.get_logger().info(f"Selected region for massage: {self.selected_region}")

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
        "--z-min", type=float, default=-0.03,
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
    parser.add_argument(
        "--model-path", type=str, 
        default="/back_massage_bot/src/back_massage_bot/models/third_runs_the_charm/weights/best.pt",
        help="Path to YOLO model weights"
    )
    parser.add_argument(
        "--yolo-low-conf", type=float, default=0.1,
        help="Lower confidence threshold for YOLO fallback detection"
    )
    parser.add_argument(
        "--point-stride", type=int, default=5,
        help="Stride for point numbering visualization (default: 5)"
    )
    parser.add_argument(
        "--massage-gun-tip-x", type=float, default=-0.09,
        help="X component of massage gun tip transform (default: 0.0)"
    )
    parser.add_argument(
        "--massage-gun-tip-y", type=float, default=0.0,
        help="Y component of massage gun tip transform (default: 0.0)"
    )
    parser.add_argument(
        "--massage-gun-tip-z", type=float, default=0.2213,
        help="Z component of massage gun tip transform (default: 0.05)"
    )
    parser.add_argument(
        "--no-visualize-path", action="store_true",
        help="Disable visualization of massage gun tip path"
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
        save_grids=args.save_grids,
        model_path=args.model_path,
        yolo_low_conf_threshold=args.yolo_low_conf,
        point_stride=args.point_stride,
        massage_gun_tip_transform=[args.massage_gun_tip_x, args.massage_gun_tip_y, args.massage_gun_tip_z],
        visualize_path=not args.no_visualize_path
    )
    wait_for_shutdown()


if __name__ == "__main__":
    main()
