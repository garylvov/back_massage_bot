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


class PointCloudTransformerAndOccupancyMapper:
    """
    Transforms point clouds and creates occupancy grids for massage planning.
    
    This class handles:
    1. Point cloud processing and cropping
    2. Occupancy grid creation
    3. YOLO-based body part detection
    4. Visualization of detections and occupancy grid
    """
    
    def __init__(self, node, robot_base_frame, crop_bounds, grid_resolution, model_path=None):
        """
        Initialize the point cloud transformer and occupancy mapper.
        
        Args:
            node: ROS2 node
            robot_base_frame: Frame ID for the robot base
            crop_bounds: Dictionary with x_min, x_max, y_min, y_max, z_min, z_max for cropping
            grid_resolution: Resolution of the occupancy grid in meters
            model_path: Path to YOLO model file (optional)
        """
        self.node = node
        self.robot_base_frame = robot_base_frame
        self.crop_bounds = crop_bounds
        self.grid_resolution = grid_resolution
        self.frame_counter = 0
        
        # Create publishers for visualization
        qos = QoSProfile(depth=10)
        self.cloud_publisher = self.node.create_publisher(PointCloud2, "/massage_planning/cropped_cloud", qos)
        self.segmented_cloud_publisher = self.node.create_publisher(PointCloud2, "/massage_planning/segmented_cloud", qos)
        self.marker_publisher = self.node.create_publisher(MarkerArray, "/massage_planning/occupancy_grid", qos)
        self.detection_publisher = self.node.create_publisher(MarkerArray, "/massage_planning/detections", qos)
        
        # Load YOLO model if path is provided
        self.model = None
        if model_path:
            try:
                self.node.get_logger().info(f"Loading YOLO model from {model_path}")
                self.model = ultralytics.YOLO(model_path)
                self.model.args['data'] = "data/data.yaml"
                self.node.get_logger().info("YOLO model loaded successfully")
            except Exception as e:
                self.node.get_logger().error(f"Failed to load YOLO model: {str(e)}")
    
    def point_cloud_to_numpy(self, cloud_msg):
        """
        Convert ROS PointCloud2 message to numpy array.
        
        Args:
            cloud_msg: ROS PointCloud2 message
            
        Returns:
            Numpy array of points with shape (N, 3) or None if conversion fails
        """
        try:
            # Convert to numpy array
            points = np.array(list(pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)))
            
            if len(points) == 0:
                self.node.get_logger().warn("Converted point cloud is empty")
                return None
                
            return points
        except Exception as e:
            self.node.get_logger().error(f"Error converting point cloud to numpy: {str(e)}")
            return None
    
    def crop_point_cloud(self, points):
        """
        Crop point cloud to specified bounds.
        
        Args:
            points: Numpy array of points with shape (N, 3)
            
        Returns:
            Cropped numpy array of points or None if all points are filtered out
        """
        try:
            # Apply crop bounds
            mask = (
                (points[:, 0] >= self.crop_bounds["x_min"]) & 
                (points[:, 0] <= self.crop_bounds["x_max"]) &
                (points[:, 1] >= self.crop_bounds["y_min"]) & 
                (points[:, 1] <= self.crop_bounds["y_max"]) &
                (points[:, 2] >= self.crop_bounds["z_min"]) & 
                (points[:, 2] <= self.crop_bounds["z_max"])
            )
            
            cropped_points = points[mask]
            
            if len(cropped_points) == 0:
                self.node.get_logger().warn("All points were filtered out by crop bounds")
                return None
                
            self.node.get_logger().info(f"Cropped point cloud from {len(points)} to {len(cropped_points)} points")
            return cropped_points
        except Exception as e:
            self.node.get_logger().error(f"Error cropping point cloud: {str(e)}")
            return None

    def create_top_down_occupancy(self, points):
        """
        Create a 2D top-down occupancy grid from the point cloud.
        
        Args:
            points: Numpy array of points with shape (N, 3)
            
        Returns:
            Dictionary with grid data including:
            - grid: 2D numpy array representing the occupancy grid
            - mask: 2D boolean numpy array of occupied cells
            - origin: (x, y) coordinates of the grid origin
            - points: 3D representative points for each grid cell
            - image: RGB image representation of the grid
        """
        try:
            # Extract x, y, z coordinates
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            z_coords = points[:, 2]
            
            # Find min and max coordinates for grid bounds
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            # Calculate grid size
            grid_width = int(np.ceil((x_max - x_min) / self.grid_resolution))
            grid_height = int(np.ceil((y_max - y_min) / self.grid_resolution))
            
            self.node.get_logger().info(f"Creating occupancy grid with dimensions {grid_width}x{grid_height}")
            
            # Initialize grid and representative points array
            grid = np.zeros((grid_width, grid_height), dtype=np.int32)
            mask = np.zeros((grid_width, grid_height), dtype=bool)
            representative_points = np.empty((grid_width, grid_height), dtype=object)
            
            # Fill grid cells
            for point in points:
                x, y, z = point
                
                # Calculate grid indices
                x_idx = int((x - x_min) / self.grid_resolution)
                y_idx = int((y - y_min) / self.grid_resolution)
                
                # Ensure indices are within bounds
                if 0 <= x_idx < grid_width and 0 <= y_idx < grid_height:
                    grid[x_idx, y_idx] += 1
                    mask[x_idx, y_idx] = True
                    
                    # Store a representative 3D point for this cell
                    representative_points[x_idx, y_idx] = point
            
            # Create an RGB image from the grid for visualization
            grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            # Fill the image - white for occupied cells
            for i in range(grid_width):
                for j in range(grid_height):
                    if mask[i, j]:
                        # Flip y-axis for image (origin is top-left in images)
                        grid_image[grid_height-1-j, i] = [255, 255, 255]
            
            # Return grid data
            return {
                "grid": grid,
                "mask": mask,
                "origin": (x_min, y_min),
                "points": representative_points,
                "image": grid_image
            }
            
        except Exception as e:
            self.node.get_logger().error(f"Error creating occupancy grid: {str(e)}")
            import traceback
            self.node.get_logger().error(traceback.format_exc())
            return None

    def create_grid_markers(self, grid, origin):
        """
        Create visualization markers for the occupancy grid.
        
        Args:
            grid: 2D numpy array representing the occupancy grid
            origin: (x, y) coordinates of the grid origin
            
        Returns:
            MarkerArray containing visualization markers
        """
        marker_array = MarkerArray()
        
        # Create point cloud marker for occupied cells
        marker = Marker()
        marker.header.frame_id = self.robot_base_frame
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "occupancy_grid"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.01  # Point size
        marker.scale.y = 0.01
        marker.color.a = 0.5  # Semi-transparent
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        
        # Add points for each occupied cell
        points = []
        x_origin, y_origin = origin
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j]:
                    x = x_origin + (i + 0.5) * self.grid_resolution
                    y = y_origin + (j + 0.5) * self.grid_resolution
                    z = self.crop_bounds["z_min"] + 0.45  # Move up by 45cm for visibility
                    
                    point = Point()
                    point.x = x
                    point.y = y
                    point.z = z
                    points.append(point)
        
        marker.points = points
        marker_array.markers.append(marker)
        
        # Create an outline of the grid area
        outline = Marker()
        outline.header.frame_id = self.robot_base_frame
        outline.header.stamp = self.node.get_clock().now().to_msg()
        outline.ns = "occupancy_grid"
        outline.id = 1
        outline.type = Marker.LINE_STRIP
        outline.action = Marker.ADD
        outline.scale.x = 0.005  # Line width
        outline.color.a = 1.0
        outline.color.r = 1.0
        outline.color.g = 1.0
        outline.color.b = 1.0
        outline.pose.orientation.w = 1.0
        
        # Add the four corners of the grid
        width = grid.shape[0] * self.grid_resolution
        height = grid.shape[1] * self.grid_resolution
        z = self.crop_bounds["z_min"] + 0.001  # Slightly above min z
        
        # Move the outline up by 45cm to match the grid
        z_elevated = z + 0.45
        
        corners = [
            (x_origin, y_origin, z_elevated),
            (x_origin + width, y_origin, z_elevated),
            (x_origin + width, y_origin + height, z_elevated),
            (x_origin, y_origin + height, z_elevated),
            (x_origin, y_origin, z_elevated)  # Close the loop
        ]
        
        # Create points for each corner
        outline_points = []
        for x, y, z in corners:
            point = Point()
            point.x = x
            point.y = y
            point.z = z
            outline_points.append(point)
        
        outline.points = outline_points
        marker_array.markers.append(outline)
        
        return marker_array

    def create_detection_debug_markers(self, grid_data, yolo_result):
        """
        Create debug visualization showing both occupancy grid and detection boxes.
        
        Args:
            grid_data: Dictionary with grid data
            yolo_result: YOLO detection results
            
        Returns:
            MarkerArray containing visualization markers
        """
        marker_array = MarkerArray()
        
        # Extract grid data
        mask = grid_data["mask"]
        grid_width, grid_height = mask.shape
        x_origin, y_origin = grid_data["origin"]
        
        # Create a marker for the grid
        grid_marker = Marker()
        grid_marker.header.frame_id = self.robot_base_frame
        grid_marker.header.stamp = self.node.get_clock().now().to_msg()
        grid_marker.ns = "debug_visualization"
        grid_marker.id = 0
        grid_marker.type = Marker.POINTS
        grid_marker.action = Marker.ADD
        grid_marker.scale.x = 0.01  # Point size
        grid_marker.scale.y = 0.01
        grid_marker.scale.z = 0.001  # Very thin in z-direction
        grid_marker.color.a = 0.3  # Semi-transparent
        grid_marker.color.r = 0.7
        grid_marker.color.g = 0.7
        grid_marker.color.b = 0.7
        grid_marker.pose.orientation.w = 1.0
        
        # Use the actual z-height from the point cloud, but move it up for better visibility
        grid_z_height = self.crop_bounds["z_min"] + 0.45  # Move grid up by 45cm
        
        # Add occupied cells
        for i in range(grid_width):
            for j in range(grid_height):
                if mask[i, j]:
                    x = x_origin + (i + 0.5) * self.grid_resolution
                    y = y_origin + (j + 0.5) * self.grid_resolution
                    
                    point = Point()
                    point.x = x
                    point.y = y
                    point.z = grid_z_height
                    grid_marker.points.append(point)
        
        marker_array.markers.append(grid_marker)
        
        # Create markers for each detection box
        if yolo_result is not None:
            # Sort boxes by confidence to get the highest confidence detection for each class
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
            
            # Process only the highest confidence detection for each class
            for cls, (box, conf) in boxes_by_class.items():
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                
                # Get original image dimensions from result
                if hasattr(yolo_result, 'orig_shape'):
                    original_height, original_width = yolo_result.orig_shape
                else:
                    original_height, original_width = grid_height, grid_width
                
                # Get YOLO dimensions - use the same logic as in get_points_in_detection
                if hasattr(box, 'orig_shape'):
                    yolo_height, yolo_width = box.orig_shape
                else:
                    # Fallback to default YOLO dimensions if not available
                    yolo_width, yolo_height = 640, 640  # Default YOLO v8 dimensions
                
                # Calculate scaling factors
                x_scale = original_width / yolo_width
                y_scale = original_height / yolo_height
                
                # Scale YOLO coordinates to original image coordinates
                x1_img = int(xyxy[0] * x_scale)
                x2_img = int(xyxy[2] * x_scale)
                y1_img = int(xyxy[1] * y_scale)
                y2_img = int(xyxy[3] * y_scale)
                
                # Convert to grid coordinates (flipping y-axis)
                x1_idx = x1_img
                x2_idx = x2_img
                y1_idx = grid_height - y2_img  # Note the swap of y2 and y1 due to flipping
                y2_idx = grid_height - y1_img
                
                # Ensure indices are within bounds
                x1_idx = max(0, min(grid_width - 1, x1_idx))
                x2_idx = max(0, min(grid_width - 1, x2_idx))
                y1_idx = max(0, min(grid_height - 1, y1_idx))
                y2_idx = max(0, min(grid_height - 1, y2_idx))
                
                # Convert to world coordinates
                x1_world = x_origin + x1_idx * self.grid_resolution
                y1_world = y_origin + y1_idx * self.grid_resolution
                x2_world = x_origin + (x2_idx + 1) * self.grid_resolution
                y2_world = y_origin + (y2_idx + 1) * self.grid_resolution
                
                # Log the detection box coordinates for debugging
                self.node.get_logger().info(f"Debug marker for detection class={cls}, conf={conf:.2f}, grid=[{x1_idx},{y1_idx},{x2_idx},{y2_idx}], world=[{x1_world:.2f},{y1_world:.2f},{x2_world:.2f},{y2_world:.2f}]")
                
                # Create box marker
                box_marker = Marker()
                box_marker.header.frame_id = self.robot_base_frame
                box_marker.header.stamp = self.node.get_clock().now().to_msg()
                box_marker.ns = "debug_visualization"
                box_marker.id = cls + 1  # Use class as ID to avoid duplicates
                box_marker.type = Marker.LINE_STRIP
                box_marker.action = Marker.ADD
                box_marker.scale.x = 0.005  # Line width
                box_marker.color.a = 1.0
                
                # Set color based on class
                colors = [
                    (1.0, 0.0, 0.0),  # Red - Torso
                    (0.0, 1.0, 0.0),  # Green - Head
                    (0.7, 0.5, 0.3)   # Brown - Legs
                ]
                color_idx = 0 if cls == 0 else (1 if cls == 1 else 2)
                color = colors[color_idx]
                box_marker.color.r = color[0]
                box_marker.color.g = color[1]
                box_marker.color.b = color[2]
                box_marker.pose.orientation.w = 1.0
                
                # Create the box outline - keep at actual height
                z = self.crop_bounds["z_min"] + 0.001  # Slightly above the min z
                corners = [
                    (x1_world, y1_world, z),
                    (x2_world, y1_world, z),
                    (x2_world, y2_world, z),
                    (x1_world, y2_world, z),
                    (x1_world, y1_world, z)  # Close the loop
                ]
                
                box_marker.points = [Point(x=x, y=y, z=z) for x, y, z in corners]
                marker_array.markers.append(box_marker)
                
                # Add text marker with class name
                text_marker = Marker()
                text_marker.header.frame_id = self.robot_base_frame
                text_marker.header.stamp = self.node.get_clock().now().to_msg()
                text_marker.ns = "debug_visualization"
                text_marker.id = cls + 100  # Offset to avoid ID collision
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.scale.z = 0.05  # Text height
                text_marker.color.a = 1.0
                text_marker.color.r = color[0]
                text_marker.color.g = color[1]
                text_marker.color.b = color[2]
                
                # Position text above the box
                text_marker.pose.position.x = (x1_world + x2_world) / 2
                text_marker.pose.position.y = (y1_world + y2_world) / 2
                text_marker.pose.position.z = z + 0.05  # Above the box
                text_marker.pose.orientation.w = 1.0
                
                # Get class name
                class_names = {
                    0: "Torso",
                    1: "Head",
                    6: "Legs"
                }
                text_marker.text = class_names.get(cls, f"Class {cls}")
                marker_array.markers.append(text_marker)
        
        return marker_array

    def create_detection_markers(self, points_by_class, class_colors):
        """
        Create marker array for visualization of detected body regions.
        
        Args:
            points_by_class: Dictionary mapping class IDs to point arrays
            class_colors: Dictionary mapping class IDs to RGB color tuples
            
        Returns:
            MarkerArray containing visualization markers for each body region
        """
        marker_array = MarkerArray()
        
        # Only include head (1), torso (0), and legs (6)
        allowed_classes = {0, 1, 6}
        
        # If we have multiple instances of the same class, keep only one
        filtered_points_by_class = {}
        
        for cls, points in points_by_class.items():
            # Skip classes we don't want to visualize
            if cls not in allowed_classes or points is None or len(points) == 0:
                continue
            
            # If we already have this class, skip it
            if cls in filtered_points_by_class:
                continue
                
            # Add this class to our filtered set
            filtered_points_by_class[cls] = points
        
        # Now create markers for the filtered set
        for cls, points in filtered_points_by_class.items():
            # Get color for this class
            color = class_colors.get(cls, (0.5, 0.5, 0.5))  # Default gray if class not found
            
            # Create point cloud marker
            marker = Marker()
            marker.header.frame_id = self.robot_base_frame
            marker.header.stamp = self.node.get_clock().now().to_msg()
            marker.ns = "body_regions"
            marker.id = cls
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.scale.x = 0.02  # Point size
            marker.scale.y = 0.02
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            
            # Add all points for this class - keep at actual height
            for point in points:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = point[2]  # Keep at actual height
                marker.points.append(p)
            
            marker_array.markers.append(marker)
            
            # Add text marker with class name
            text_marker = Marker()
            text_marker.header.frame_id = self.robot_base_frame
            text_marker.header.stamp = self.node.get_clock().now().to_msg()
            text_marker.ns = "body_regions"
            text_marker.id = cls + 100  # Offset to avoid ID collision
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.scale.z = 0.05  # Text height
            text_marker.color.a = 1.0
            text_marker.color.r = color[0]
            text_marker.color.g = color[1]
            text_marker.color.b = color[2]
            
            # Calculate centroid of points for text placement
            if len(points) > 0:
                centroid = np.mean(points, axis=0)
                text_marker.pose.position.x = centroid[0]
                text_marker.pose.position.y = centroid[1]
                text_marker.pose.position.z = centroid[2] + 0.1  # Above the points
                text_marker.pose.orientation.w = 1.0
                
                # Get class name
                class_names = {
                    0: "Torso",
                    1: "Head",
                    6: "Legs"
                }
                text_marker.text = class_names.get(cls, f"Class {cls}")
                marker_array.markers.append(text_marker)
        
        return marker_array

    def get_points_in_detection(self, points, mask, representative_points, origin, detection_box):
        """
        Extract points that fall within a detection bounding box.
        
        Args:
            points: Numpy array of points with shape (N, 3)
            mask: 2D boolean numpy array of occupied cells
            representative_points: 2D array of representative points for each cell
            origin: (x, y) coordinates of the grid origin
            detection_box: YOLO detection box
            
        Returns:
            Numpy array of points within the detection box or None if no points found
        """
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
            
            # Instead of using the grid mask, directly filter the original points
            # This ensures perfect alignment with the cropped cloud
            detection_points = []
            for point in points:
                x, y, z = point
                # Check if the point is within the detection box in world coordinates
                if (x1_world <= x <= x2_world and 
                    y1_world <= y <= y2_world):
                    detection_points.append(point)
            
            self.node.get_logger().info(f"Found {len(detection_points)} points in detection region")
            return np.array(detection_points) if detection_points else None
            
        except Exception as e:
            self.node.get_logger().error(f"Error extracting points for detection: {str(e)}")
            import traceback
            self.node.get_logger().error(traceback.format_exc())
            return None

    def run_yolo_inference(self, grid_image):
        """
        Run YOLO inference directly on the grid image numpy array.
        
        Args:
            grid_image: RGB image of the occupancy grid as numpy array
            
        Returns:
            YOLO detection results or None if inference fails
        """
        try:
            if self.model is None:
                self.node.get_logger().warn("YOLO model not loaded, skipping inference")
                return None
                
            # Run inference directly on the numpy array
            self.node.get_logger().info(f"Running YOLO inference on grid image array")
            results = self.model.predict(
                grid_image,
                conf=0.25,  # Confidence threshold
                verbose=False
            )
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                self.node.get_logger().info(f"YOLO detected {len(results[0].boxes)} objects")
                return results[0]
            else:
                self.node.get_logger().warn("YOLO did not detect any objects, trying with lower confidence")
                
                # Try again with lower confidence threshold
                results = self.model.predict(
                    grid_image,
                    conf=0.1,  # Lower confidence threshold
                    verbose=False
                )
                
                if len(results) > 0 and len(results[0].boxes) > 0:
                    self.node.get_logger().info(f"YOLO detected {len(results[0].boxes)} objects with lower threshold")
                    return results[0]
                
                self.node.get_logger().warn("YOLO did not detect any objects even with lower threshold")
                return None
                
        except Exception as e:
            self.node.get_logger().error(f"Error running YOLO inference: {str(e)}")
            import traceback
            self.node.get_logger().error(traceback.format_exc())
            return None

    def point_cloud_callback(self, msg):
        """
        Process incoming point cloud messages.
        
        Args:
            msg: ROS PointCloud2 message
        """
        try:
            # Convert point cloud message to numpy array
            points = self.point_cloud_to_numpy(msg)
            if points is None:
                self.node.get_logger().warn("Failed to convert point cloud to numpy")
                return
            
            # Apply crop bounds to focus on the massage area
            cropped_points = self.crop_point_cloud(points)
            if cropped_points is None:
                self.node.get_logger().warn("All points were filtered out by crop bounds")
                return
            
            # Create occupancy grid from cropped point cloud
            grid_data = self.create_top_down_occupancy(cropped_points)
            if grid_data is None:
                self.node.get_logger().error("Failed to create occupancy grid")
                return
            
            # Publish occupancy grid visualization
            marker_array = self.create_grid_markers(grid_data["mask"], grid_data["origin"])
            self.marker_publisher.publish(marker_array)
            self.node.get_logger().info(f"Published occupancy grid with {np.sum(grid_data['mask'])} occupied cells")
            
            # Publish the full cropped point cloud (before segmentation)
            header = msg.header
            header.frame_id = self.robot_base_frame
            cropped_cloud_msg = pc2.create_cloud_xyz32(header, cropped_points)
            self.cloud_publisher.publish(cropped_cloud_msg)
            self.node.get_logger().info(f"Published cropped point cloud with {len(cropped_points)} points")
            
            # Run YOLO inference directly on the grid image
            if self.model is not None:
                yolo_result = self.run_yolo_inference(grid_data["image"])
                
                if yolo_result is not None and len(yolo_result.boxes) > 0:
                    # Create debug visualization showing grid and detection boxes
                    debug_topic = "/massage_planning/debug_visualization"
                    debug_publisher = self.node.create_publisher(MarkerArray, debug_topic, 10)
                    debug_markers = self.create_detection_debug_markers(grid_data, yolo_result)
                    debug_publisher.publish(debug_markers)
                    self.node.get_logger().info(f"Published debug visualization to {debug_topic}")
                    
                    # Define colors for each class (RGB format)
                    class_colors = {
                        0: (0.9, 0.2, 0.2),  # Torso - red
                        1: (0.2, 0.9, 0.2),  # Head - green
                        6: (0.7, 0.5, 0.3)   # legs - brown
                    }
                    
                    # Filter boxes to get the highest confidence detection for each class
                    # ONLY include classes 0 (torso), 1 (head), and 6 (legs)
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
                        detection_markers = self.create_detection_markers(points_by_class, class_colors)
                        if detection_markers.markers:
                            self.detection_publisher.publish(detection_markers)
                            self.node.get_logger().info(f"Published detection markers with {len(detection_markers.markers)} regions")
                        else:
                            self.node.get_logger().warn("No detection markers created")
                    else:
                        self.node.get_logger().warn("No points found in detection regions")
                    
                    # Publish the segmented point cloud (only head, torso, legs)
                    filtered_points = []
                    if points_by_class:
                        for cls, points in points_by_class.items():
                            if cls in [0, 1, 6]:
                                filtered_points.extend(points)
                    
                    if filtered_points:
                        filtered_points = np.array(filtered_points)
                        filtered_cloud_msg = pc2.create_cloud_xyz32(header, filtered_points)
                        self.segmented_cloud_publisher.publish(filtered_cloud_msg)
                        self.node.get_logger().info(f"Published segmented point cloud with {len(filtered_points)} points")
                    else:
                        self.node.get_logger().warn("No points belong to the head, torso, or legs classes")
            
            # Increment frame counter
            self.frame_counter += 1
            
        except Exception as e:
            self.node.get_logger().error(f"Error processing point cloud: {str(e)}")
            import traceback
            self.node.get_logger().error(traceback.format_exc())

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
        "--occupancy-topic", type=str, default="/massage_planning/occupancy_grid",
        help="Output topic for the occupancy grid visualization"
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
        "--z-min", type=float, default=0.05,
        help="Minimum Z value (in robot base frame) for cropping"
    )
    parser.add_argument(
        "--z-max", type=float, default=0.5,
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
    node = ros_scope.node()
    robot_base_frame = args.robot_base_frame
    crop_bounds = {
        "x_min": args.x_min,
        "x_max": args.x_max,
        "y_min": args.y_min,
        "y_max": args.y_max,
        "z_min": args.z_min,
        "z_max": args.z_max,
    }
    grid_resolution = args.grid_resolution
    model_path = "/back_massage_bot/src/back_massage_bot/models/third_runs_the_charm/weights/best.pt"
    
    PointCloudTransformerAndOccupancyMapper(
        node, 
        robot_base_frame, 
        crop_bounds, 
        grid_resolution, 
        model_path
    )
    wait_for_shutdown()


if __name__ == "__main__":
    main()