#!/usr/bin/env python3

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
import os

def create_coordinate_frame(transform, size=0.05, label=None):
    """Create a coordinate frame mesh fromt a 4x4 transform matrix."""
    geometries = []
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(transform)
    geometries.append(frame)
    
    if label is not None:
        # Create text using TriangleMesh - make it much smaller
        text = o3d.t.geometry.TriangleMesh.create_text(str(label), depth=size/500).to_legacy()
        text.paint_uniform_color([1, 0, 0])  # Red color for visibility
        
        # Create three text labels rotated 120 degrees apart for better visibility
        for angle in [0, 120, 240]:
            # Create a copy of the text mesh for each orientation
            text_copy = o3d.geometry.TriangleMesh()
            text_copy.vertices = o3d.utility.Vector3dVector(np.asarray(text.vertices))
            text_copy.triangles = o3d.utility.Vector3iVector(np.asarray(text.triangles))
            text_copy.vertex_colors = o3d.utility.Vector3dVector(np.asarray(text.vertex_colors))
            
            # Create rotation matrix around Z axis
            rotation = Rotation.from_euler("Z", angle, degrees=True).as_matrix()
            
            # Scale and position text above coordinate frame - reduced scale significantly
            text_transform = np.array([
                [size/100, 0, 0, transform[0, 3]],
                [0, size/100, 0, transform[1, 3]],
                [0, 0, size/100, transform[2, 3] + size/10],
                [0, 0, 0, 1]
            ])
            
            # Apply rotation to the text transform
            text_transform[:3, :3] = text_transform[:3, :3] @ rotation
            text_copy.transform(text_transform)
            geometries.append(text_copy)
    
    return geometries

def plan_massage_from_points(pcd, 
                             stride=1, 
                             point_normal_threshold_degrees=30,
                             tip_offset=[0, 0, -.254],
                             gravity_compensation_angle=25,
                             region_name=None):
    """
    
    tip_offset=[.0765, 0, -.254]
    
    Generate transforms from a list of points with stride-based sampling.
    Only includes points where the normal is less than 45 degrees from vertical.
    Args:
        pcd: Open3D point cloud
        stride (int): Number of points to skip between samples
        ow 
    Returns:
        list: List of tuples (transform_matrix, label) where label is the sequence number
    """
    print("TIP OFFSET: ", tip_offset)
    points = np.asarray(pcd.points)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(k=6) 
    normals = np.asarray(pcd.normals)
    
    base_transform = np.eye(4, dtype=np.float64)
    base_transform[:3, :3] = Rotation.from_euler("X", 180, degrees=True).as_matrix()
    base_transform_2 = np.eye(4, dtype=np.float64)
    base_transform_2[:3, :3] = Rotation.from_euler("Z", -90, degrees=True).as_matrix()
    to_tip_transform = np.eye(4, dtype=np.float64)
    to_tip_transform[:3, 3] = tip_offset
   
    if "left" in region_name:
        to_tip_transform[:3, 3] = np.array([0, tip_offset[0], tip_offset[2]])
        dumb_gravity_comp = np.eye(4, dtype=np.float64)
        dumb_gravity_comp[:3, :3] = Rotation.from_euler("Y", -gravity_compensation_angle, degrees=True).as_matrix()
        print("Warn; doing niave gravity comp for left side")
        transform_chain_to_align = [base_transform, base_transform_2, to_tip_transform, dumb_gravity_comp]
    
    if "right" in region_name:
        transform_chain_to_align = [base_transform, to_tip_transform]
    
    # First collect all valid transforms based on normal angle
    vertical = np.array([0, 0, 1])
    angle_threshold = np.radians(point_normal_threshold_degrees)
    valid_transforms = []
    
    for i in range(len(points)):
        # Check normal angle
        angle = np.arccos(np.abs(np.dot(normals[i], vertical)))
        if angle >= angle_threshold:
            continue
            
        # Create transform for valid point
        unit_point_transform = np.eye(4, dtype=np.float64)
        unit_point_transform[:3, 3] = points[i]
        
        final_transform = unit_point_transform
        for transform in transform_chain_to_align:
            final_transform = final_transform @ transform
            
        valid_transforms.append((final_transform, i + 1))
    
    # Then apply stride filtering
    transforms = valid_transforms[::stride]
    
    # Renumber the transforms sequentially
    for i, (transform, _) in enumerate(transforms):
        transforms[i] = (transform, i + 1)
    
    return transforms

def visualize_pcd_with_transforms(pcd_path, transforms=None):
    
    """
    Visualize a PCD file with optional coordinate frames and labels.
    
    Args:
        pcd_path (str): Path to the PCD file
        transforms (list): List of tuples (transform_matrix, label) to visualize
    """
    # Read and prepare the PCD file
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(k=6)
    
    # Create visualization geometries
    geometries = [pcd]
    
    # Add coordinate frames if transforms are provided
    if transforms is not None:
        for transform, label in transforms:
            geometries.extend(create_coordinate_frame(transform, label=label))
    
    # Visualize with normals
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add geometries
    for geometry in geometries:
        vis.add_geometry(geometry)
    
    # Set default camera view
    vis.get_view_control().set_zoom(0.8)
    
    # Enable normal rendering
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
    opt.point_show_normal = True  # Show normals
    opt.light_on = True  # Enable lighting
    
    # Run visualization
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # Step 1: Load cloud
    pcd_path = "region_left_middle_20250401_045133.pcd"
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    # Step 2: Find transforms with stride
    stride = 1  # Adjust this value to change sampling density
    transforms = plan_massage_from_points(pcd, stride=stride)
    
    # Add origin frame as the last numbered transform
    next_number = len(transforms) + 1
    transforms.append((np.eye(4, dtype=np.float64), next_number))
    
    # Step 3: Visualize
    visualize_pcd_with_transforms(pcd_path, transforms) 