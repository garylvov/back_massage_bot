#!/usr/bin/env python3

import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

def create_detection_debug_markers(grid_data, yolo_result, robot_base_frame, grid_resolution, z_min, logger=None):
    """Create debug visualization showing both occupancy grid and detection boxes"""
    marker_array = MarkerArray()
    
    # Extract grid data
    mask = grid_data["mask"]
    grid_width, grid_height = mask.shape
    x_origin, y_origin = grid_data["origin"]
    
    # Create a marker for the grid
    grid_marker = Marker()
    grid_marker.header.frame_id = robot_base_frame
    # Note: timestamp will be set by the caller
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
    z_height = z_min
    
    # Add occupied cells
    for i in range(grid_width):
        for j in range(grid_height):
            if mask[i, j]:
                x = x_origin + (i + 0.5) * grid_resolution
                y = y_origin + (j + 0.5) * grid_resolution
                
                point = Point()
                point.x = x
                point.y = y
                point.z = z_height
                grid_marker.points.append(point)
    
    marker_array.markers.append(grid_marker)
    
    # Create an outline of the grid area
    outline = Marker()
    outline.header.frame_id = robot_base_frame
    # Note: timestamp will be set by the caller
    outline.ns = "debug_visualization"
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
    width = grid_width * grid_resolution
    height = grid_height * grid_resolution
    z = z_height + 0.001  # Slightly above min z
    
    corners = [
        (x_origin, y_origin, z),
        (x_origin + width, y_origin, z),
        (x_origin + width, y_origin + height, z),
        (x_origin, y_origin + height, z),
        (x_origin, y_origin, z)  # Close the loop
    ]
    
    outline.points = [Point(x=x, y=y, z=z) for x, y, z in corners]
    marker_array.markers.append(outline)
    
    # Create markers for each detection box
    if yolo_result is not None:
        # Track which classes we've already processed to avoid duplicates
        processed_classes = set()
        
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
            x1_world = x_origin + x1_idx * grid_resolution
            y1_world = y_origin + y1_idx * grid_resolution
            x2_world = x_origin + (x2_idx + 1) * grid_resolution
            y2_world = y_origin + (y2_idx + 1) * grid_resolution
            
            # Log the detection box coordinates for debugging
            if logger:
                logger.info(f"Debug marker for detection class={cls}, conf={conf:.2f}, grid=[{x1_idx},{y1_idx},{x2_idx},{y2_idx}], world=[{x1_world:.2f},{y1_world:.2f},{x2_world:.2f},{y2_world:.2f}]")
            
            # Create box marker
            box_marker = Marker()
            box_marker.header.frame_id = robot_base_frame
            # Note: timestamp will be set by the caller
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
            
            # Create the box outline - moved up for better visibility
            z = z_height + 0.001  # Slightly above the grid
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
            text_marker.header.frame_id = robot_base_frame
            # Note: timestamp will be set by the caller
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

def create_detection_markers(points_by_class, class_colors, robot_base_frame, logger=None):
    """Create marker array for visualization of detected regions"""
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
        marker.header.frame_id = robot_base_frame
        # Note: timestamp will be set by the caller
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
        
        # Add all points for this class
        for point in points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            # Move points up for better visibility
            p.z = point[2]
            marker.points.append(p)
        
        marker_array.markers.append(marker)
        
        # Add text marker with class name
        text_marker = Marker()
        text_marker.header.frame_id = robot_base_frame
        # Note: timestamp will be set by the caller
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
            text_marker.pose.position.z = centroid[2] + 0.5  # Above the points
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