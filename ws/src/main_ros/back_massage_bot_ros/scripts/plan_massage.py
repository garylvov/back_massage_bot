#!/usr/bin/env python3

# Copyright (c) 2025, Gary Lvov, Vinay Balaji, Tim Bennet, Xandar Ingare, Ben Yoon
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
A ROS2 node for transforming and cropping point clouds from camera to robot frame.
Subscribes to a point cloud topic, transforms points to the robot base frame using Open3D,
crops based on specified bounds, and creates an occupancy grid with 1cm resolution.
"""

import argparse
import os
import sys
import numpy as np
import open3d as o3d
from datetime import datetime
from typing import Optional, Tuple, List
import struct
import matplotlib.pyplot as plt

import synchros2.process as ros_process
import synchros2.scope as ros_scope
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped
from synchros2.context import wait_for_shutdown
from synchros2.utilities import namespace_with
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point


class PointCloudTransformerAndOccupancyMapper:
    def __init__(
        self,
        input_topic: str = "/depth/color/points",
        output_topic: str = "/massage_planning/cropped_cloud",
        occupancy_topic: str = "/massage_planning/occupancy_grid",
        camera_frame: str = "camera_depth_optical_frame",
        robot_base_frame: str = "j2n6s300_link_base",
        tf_prefix: Optional[str] = None,
        x_min: float = -0.7,
        x_max: float = 1.4,
        y_min: float = -0.8,
        y_max: float = -0.1,
        z_min: float = 0.0,
        z_max: float = 1.0,
        grid_resolution: float = 0.01,  # 1cm grid resolution
        output_dir: str = "~/pointcloud_plys",
        save_plys: bool = False,
        save_grids: bool = True
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
        
        # Create a unique directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.output_dir, f"run_{timestamp}")
        self.grid_dir = os.path.join(self.run_dir, "grid_images")
        
        # Create output directories
        if self.save_plys or self.save_grids:
            os.makedirs(self.run_dir, exist_ok=True)
            self.node.get_logger().info(f"Created output directory at {self.run_dir}")
            
            if self.save_grids:
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
        
        # Cache for the transform - only look it up once
        self.transform_matrix = None
        self.tried_transform_lookup = False
        
        # Frame counter for grid images
        self.frame_counter = 0
        
        # Subscribe to the point cloud topic
        self.subscription = self.node.create_subscription(
            PointCloud2,
            self.input_topic,
            self.point_cloud_callback,
            qos
        )
        
        self.node.get_logger().info(f"Subscribed to {self.input_topic}")
        self.node.get_logger().info(f"Publishing transformed and cropped point cloud to {self.output_topic}")
        self.node.get_logger().info(f"Publishing occupancy grid visualization to {self.occupancy_topic}")
        self.node.get_logger().info("Waiting for point clouds...")

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

    def create_occupancy_grid(self, points):
        """Create a 2D occupancy grid from the point cloud"""
        try:
            # Calculate grid dimensions based on crop bounds
            x_min, x_max = self.crop_bounds["x_min"], self.crop_bounds["x_max"]
            y_min, y_max = self.crop_bounds["y_min"], self.crop_bounds["y_max"]
            
            # Calculate grid size
            grid_width = int(np.ceil((x_max - x_min) / self.grid_resolution))
            grid_height = int(np.ceil((y_max - y_min) / self.grid_resolution))
            
            # Create empty grid
            grid = np.zeros((grid_width, grid_height), dtype=bool)
            
            # Fill grid cells
            for point in points:
                # Calculate grid cell indices
                x_idx = int((point[0] - x_min) / self.grid_resolution)
                y_idx = int((point[1] - y_min) / self.grid_resolution)
                
                # Check if the indices are within grid bounds
                if 0 <= x_idx < grid_width and 0 <= y_idx < grid_height:
                    grid[x_idx, y_idx] = True
            
            self.node.get_logger().info(f"Created occupancy grid with dimensions {grid_width}x{grid_height}")
            
            return grid, (x_min, y_min)
        except Exception as e:
            self.node.get_logger().error(f"Error creating occupancy grid: {str(e)}")
            return None, None

    def create_grid_markers(self, grid, origin):
        """Create marker array for visualization of occupancy grid"""
        marker_array = MarkerArray()
        occupied_cells = np.where(grid)
        
        # Create a single cube marker for all cells
        marker = Marker()
        marker.header.frame_id = self.robot_base_frame
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "occupancy_grid"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.scale.x = self.grid_resolution
        marker.scale.y = self.grid_resolution
        marker.scale.z = self.grid_resolution * 0.1  # Thin in z-direction
        marker.color.a = 0.7  # Semi-transparent
        marker.color.r = 0.0
        marker.color.g = 0.7
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        
        # Add all occupied cells as points
        x_origin, y_origin = origin
        points = []
        for i, j in zip(occupied_cells[0], occupied_cells[1]):
            x = x_origin + (i + 0.5) * self.grid_resolution
            y = y_origin + (j + 0.5) * self.grid_resolution
            z = self.crop_bounds["z_min"] + self.grid_resolution * 0.05  # Place just above min z
            
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
        
        corners = [
            (x_origin, y_origin, z),
            (x_origin + width, y_origin, z),
            (x_origin + width, y_origin + height, z),
            (x_origin, y_origin + height, z),
            (x_origin, y_origin, z)  # Close the loop
        ]
        
        outline.points = [Point(x=x, y=y, z=z) for x, y, z in corners]
        marker_array.markers.append(outline)
        
        return marker_array

    def save_grid_as_image(self, grid, frame_number):
        """Save the occupancy grid as a PNG image"""
        try:
            # Create a visual representation of the grid
            # White pixels for occupied cells, black for free space
            grid_image = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
            grid_image[grid] = 255  # Set occupied cells to white
            
            # Flip the image to match the correct orientation
            grid_image = np.flipud(grid_image)
            
            # Create the filename with frame number
            filename = f"grid_{frame_number:04d}.png"
            filepath = os.path.join(self.grid_dir, filename)
            
            # Save the image using matplotlib (without any titles or axes)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid_image)
            plt.axis('off')  # Turn off axes
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove margins
            plt.savefig(filepath, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Save a text file with metadata
            meta_filepath = os.path.join(self.grid_dir, f"grid_{frame_number:04d}_meta.txt")
            with open(meta_filepath, 'w') as f:
                f.write(f"Grid Resolution: {self.grid_resolution} meters\n")
                f.write(f"Grid Dimensions: {grid.shape[0]}x{grid.shape[1]} cells\n")
                f.write(f"Occupied Cells: {np.sum(grid)}\n")
                f.write(f"X Range: {self.crop_bounds['x_min']} to {self.crop_bounds['x_max']}\n")
                f.write(f"Y Range: {self.crop_bounds['y_min']} to {self.crop_bounds['y_max']}\n")
                f.write(f"Z Range: {self.crop_bounds['z_min']} to {self.crop_bounds['z_max']}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Also save a raw NumPy version for easier processing later
            np_filepath = os.path.join(self.grid_dir, f"grid_{frame_number:04d}.npy")
            np.save(np_filepath, grid)
            
            self.node.get_logger().info(f"Saved grid image to {filepath}")
            return True
        except Exception as e:
            self.node.get_logger().error(f"Error saving grid image: {str(e)}")
            return False

    def point_cloud_callback(self, msg: PointCloud2) -> None:
        """Process incoming point cloud messages using Open3D"""
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
            
            # Create occupancy grid from ALL cropped points (not just bottom points)
            if len(cropped_points) > 0:
                grid, origin = self.create_occupancy_grid(cropped_points)
                
                if grid is not None:
                    # Create and publish marker array for visualization
                    marker_array = self.create_grid_markers(grid, origin)
                    self.marker_publisher.publish(marker_array)
                    self.node.get_logger().info(f"Published occupancy grid with {np.sum(grid)} occupied cells")
                    
                    # Save grid as image if requested
                    if self.save_grids:
                        self.save_grid_as_image(grid, self.frame_counter)
                        self.frame_counter += 1
            else:
                self.node.get_logger().warn("No points found for occupancy grid")
            
            # Publish the cropped point cloud
            header = msg.header
            header.frame_id = self.robot_base_frame
            
            # Create point cloud message
            cropped_cloud_msg = pc2.create_cloud_xyz32(header, cropped_points)
            self.cloud_publisher.publish(cropped_cloud_msg)
            self.node.get_logger().info(f"Published cropped point cloud with {len(cropped_points)} points")
            
            # Save PLY if requested
            if self.save_plys:
                filename = f"cropped_cloud_{self.frame_counter-1:04d}.ply"
                filepath = os.path.join(self.run_dir, filename)
                o3d.io.write_point_cloud(filepath, cropped_pcd)
                self.node.get_logger().info(f"Saved cropped point cloud to {filepath}")
                
        except Exception as e:
            self.node.get_logger().error(f"Error processing point cloud: {str(e)}")
    
    def visualize_point_cloud(self, pcd: o3d.geometry.PointCloud) -> None:
        """Visualize the point cloud using Open3D (for debugging)"""
        o3d.visualization.draw_geometries([pcd])


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