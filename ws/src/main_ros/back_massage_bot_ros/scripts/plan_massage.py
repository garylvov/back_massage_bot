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
        spine_width_fraction: float = 5.0,  # Spine width as fraction of torso width (1/5)
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

    def create_region_motion_plans(self, detailed_regions, point_stride=5, 
                                  massage_gun_tip_transform=None, visualize_path=True):
        """
        Create motion plans for each detected back region
        
        Args:
            detailed_regions: Dictionary of detailed back regions
            point_stride: Stride for point selection in motion planning
            massage_gun_tip_transform: Optional [x,y,z] translation to apply to points
            visualize_path: Whether to visualize the massage gun tip path
            
        Returns:
            Dictionary mapping region names to motion plans with position and orientation
        """
        region_motion_plans = {}
        logger = self.node.get_logger()
        
        # Create motion plans for each region
        for region_name, (region_points, region_color) in detailed_regions.items():
            # Skip the spine region for motion planning
            if region_name == "spine":
                logger.debug(f"Skipping motion planning for spine region")
                continue
            
            # Create motion plan for this region
            motion_plan = create_massage_motion_plan(
                region_points,
                stride=point_stride,
                massage_gun_tip_transform=massage_gun_tip_transform,
                logger=logger
            )
            
            # Create orientation for each point - pointing into the body (perpendicular to back)
            # In the robot base frame, the back is roughly in the XY plane with Z pointing up
            # We want the Z-axis of the tool to point into the body (negative Y direction)
            # This means:
            # - Tool Z-axis points in negative Y direction (into the body)
            # - Tool X-axis points in positive X direction (up the body)
            # - Tool Y-axis points in positive Z direction (to the right when facing the back)
            
            # This quaternion represents the rotation from the robot base frame to the tool frame
            # where the tool Z-axis points into the body (negative Y in robot frame)
            into_body_orientation = [0.7071068, 0.0, -0.7071068, 0.0]  # Quaternion [x,y,z,w] for -90° around Z
            
            # Add orientation to all points in the motion plan
            motion_plan_with_orientation = {
                'rows': [],
                'connections': [],
                'all_points': []
            }
            
            # Add orientation to each row
            for row in motion_plan['rows']:
                row_with_orientation = []
                for point in row:
                    # Each point becomes a tuple of (position, orientation)
                    row_with_orientation.append((point, into_body_orientation))
                motion_plan_with_orientation['rows'].append(row_with_orientation)
            
            # Add orientation to connections with intermediate higher points
            for start_point, end_point in motion_plan['connections']:
                # Create an intermediate point that's higher than both start and end points
                # to avoid potential collisions during transitions between rows
                intermediate_z_offset = 0.10  # 10cm higher
                
                # Find the maximum z value between start and end points
                max_z = max(start_point[2], end_point[2])
                
                # Create the intermediate point
                intermediate_point = [
                    (start_point[0] + end_point[0]) / 2.0,  # Midpoint in x
                    (start_point[1] + end_point[1]) / 2.0,  # Midpoint in y
                    max_z + intermediate_z_offset           # Higher z
                ]
                
                # Add the connection with the intermediate point
                motion_plan_with_orientation['connections'].append(
                    ((start_point, into_body_orientation), (intermediate_point, into_body_orientation))
                )
                motion_plan_with_orientation['connections'].append(
                    ((intermediate_point, into_body_orientation), (end_point, into_body_orientation))
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
                
                # Create numbered point markers with section-based numbering
                numbered_markers = create_numbered_point_markers(
                    region_points,
                    self.robot_base_frame,
                    marker_namespace=namespace,
                    stride=point_stride,
                    section_based=True,  # Enable section-based numbering
                    logger=logger,
                    massage_gun_tip_transform=massage_gun_tip_transform,
                    visualize_path=visualize_path,
                    region_color=region_color,
                    orientation=into_body_orientation  # Pass orientation to visualization
                )
                
                # Publish numbered point markers
                self.marker_publisher.publish(numbered_markers)
                logger.debug(f"Published section-based numbered point markers for region {region_name} (stride={point_stride})")
        
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
            
            # Publish the planned path to TF for visualization
            self.publish_motion_plan_to_tf(region_name, points_with_orientation)
            
            # Execute each point in the motion plan
            for i, (point, orientation) in enumerate(points_with_orientation):
                # Create a PoseStamped message
                pose_msg = PoseStamped()
                pose_msg.header.frame_id = self.robot_base_frame
                pose_msg.header.stamp = self.node.get_clock().now().to_msg()
                
                # Set position
                pose_msg.pose.position.x = point[0]
                pose_msg.pose.position.y = point[1]
                pose_msg.pose.position.z = point[2]
                
                # Set orientation
                pose_msg.pose.orientation.x = orientation[0]
                pose_msg.pose.orientation.y = orientation[1]
                pose_msg.pose.orientation.z = orientation[2]
                pose_msg.pose.orientation.w = orientation[3]
                
                # Publish the pose command
                self.arm_dispatch_pub.publish(pose_msg)
                self.node.get_logger().info(f"Moving to point {i+1}/{len(points_with_orientation)} in {region_name}")
                
                # Wait for the arm to reach the position (simple delay-based approach)
                # In a more sophisticated implementation, you would wait for feedback from the arm
                time.sleep(2.0)  # Adjust this delay based on your robot's speed
            
            return True
            
        except Exception as e:
            self.node.get_logger().error(f"Error executing motion plan: {str(e)}")
            import traceback
            self.node.get_logger().error(traceback.format_exc())
            return False

    def publish_motion_plan_to_tf(self, region_name, points_with_orientation):
        """Publish the motion plan points to TF for visualization"""
        try:
            # Create transforms for each point in the motion plan
            for i, (point, orientation) in enumerate(points_with_orientation):
                # Create a transform
                transform = TransformStamped()
                transform.header.stamp = self.node.get_clock().now().to_msg()
                transform.header.frame_id = self.robot_base_frame
                transform.child_frame_id = f"massage_point_{region_name}_{i}"
                
                # Set translation
                transform.transform.translation.x = point[0]
                transform.transform.translation.y = point[1]
                transform.transform.translation.z = point[2]
                
                # Set rotation
                transform.transform.rotation.x = orientation[0]
                transform.transform.rotation.y = orientation[1]
                transform.transform.rotation.z = orientation[2]
                transform.transform.rotation.w = orientation[3]
                
                # Publish the transform
                self.tf_broadcaster.sendTransform(transform)
            
            self.node.get_logger().debug(f"Published {len(points_with_orientation)} points to TF for region: {region_name}")
            
        except Exception as e:
            self.node.get_logger().error(f"Error publishing motion plan to TF: {str(e)}")

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
        "--massage-gun-tip-x", type=float, default=0.0,
        help="X component of massage gun tip transform (default: 0.0)"
    )
    parser.add_argument(
        "--massage-gun-tip-y", type=float, default=0.0,
        help="Y component of massage gun tip transform (default: 0.0)"
    )
    parser.add_argument(
        "--massage-gun-tip-z", type=float, default=0.15,
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