#!/usr/bin/env python3

import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import os

def convert_yolo_box_to_world_coords(box, grid_data, logger=None):
    """
    Convert YOLO detection box coordinates to world coordinates
    
    Args:
        box: YOLO detection box
        grid_data: Dictionary with grid information
        logger: Optional ROS logger for debug messages
        
    Returns:
        Tuple of (x1_world, y1_world, x2_world, y2_world)
    """
    # Extract grid data
    mask = grid_data["mask"]
    grid_width, grid_height = mask.shape
    x_origin, y_origin = grid_data["origin"]
    grid_resolution = grid_data.get("resolution", 0.01)  # Default to 1cm if not provided
    
    # If world coordinates are already provided, use them
    if hasattr(box, 'world_coords'):
        return box.world_coords
    
    # Get box coordinates from YOLO result
    xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
    
    # Get original image dimensions
    if hasattr(box, 'orig_shape'):
        original_height, original_width = box.orig_shape
    else:
        original_height, original_width = grid_height, grid_width
    
    # Get YOLO dimensions
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
    
    # Log the conversion if logger is provided
    if logger:
        logger.info(f"Converted YOLO box {xyxy} to world coords: [{x1_world:.2f},{y1_world:.2f},{x2_world:.2f},{y2_world:.2f}]")
    
    return (x1_world, y1_world, x2_world, y2_world)

def create_yolo_markers(grid_data, detections, frame_id, grid_resolution, z_height, logger=None):
    """
    Create visualization markers for the occupancy grid and YOLO detection boxes
    
    Args:
        grid_data: Dictionary with grid information
        detections: Either a YOLO result object, a dictionary of filtered detections {class_id: (box, conf)},
                   or a custom object with detections and detailed_regions attributes
        frame_id: Frame ID for the markers
        grid_resolution: Grid resolution in meters
        z_height: Height (Z) value for the grid visualization
        logger: Optional ROS logger for debug messages
    
    Returns:
        MarkerArray with visualization markers
    """
    markers = MarkerArray()
    
    # Extract grid data
    mask = grid_data["mask"]
    grid_width, grid_height = mask.shape
    x_origin, y_origin = grid_data["origin"]
    
    # Create a marker for the grid
    grid_marker = Marker()
    grid_marker.header.frame_id = frame_id
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
    
    # Add occupied cells
    for i in range(grid_width):
        for j in range(grid_height):
            if mask[i, j]:
                p = Point()
                p.x = x_origin + i * grid_resolution
                p.y = y_origin + j * grid_resolution
                p.z = z_height
                grid_marker.points.append(p)
    
    markers.markers.append(grid_marker)
    
    # Define class names for better visualization
    class_names = {
        0: "Torso",
        1: "Head",
        6: "Legs"
    }
    
    # Add markers for detection boxes
    if detections is not None:
        marker_id = 1
        
        # Handle different types of detection objects
        if hasattr(detections, 'boxes'):
            # It's a YOLO result object
            boxes = detections.boxes
            for box in boxes:
                try:
                    # Get box coordinates
                    x1_world, y1_world, x2_world, y2_world = convert_yolo_box_to_world_coords(
                        box, grid_data, logger=logger
                    )
                    
                    # Get class ID and confidence
                    class_id = int(box.cls.cpu().numpy()[0])
                    conf = float(box.conf.cpu().numpy()[0])
                    
                    # Get class name
                    class_name = class_names.get(class_id, f"Class{class_id}")
                    
                    # Log debug info
                    if logger:
                        logger.info(f"Debug marker for detection class={class_id}, conf={conf:.2f}, world=[{x1_world:.2f},{y1_world:.2f},{x2_world:.2f},{y2_world:.2f}]")
                    
                    # Create box marker
                    box_marker = Marker()
                    box_marker.header.frame_id = frame_id
                    # Note: timestamp will be set by the caller
                    box_marker.ns = "debug_visualization"
                    box_marker.id = marker_id
                    box_marker.type = Marker.LINE_STRIP
                    box_marker.action = Marker.ADD
                    box_marker.scale.x = 0.005  # Line width
                    box_marker.color.a = 1.0
                    
                    # Set color based on class
                    if class_id == 0:  # Torso
                        box_marker.color.r = 0.9
                        box_marker.color.g = 0.2
                        box_marker.color.b = 0.2
                    elif class_id == 1:  # Head
                        box_marker.color.r = 0.2
                        box_marker.color.g = 0.9
                        box_marker.color.b = 0.2
                    elif class_id == 6:  # Legs
                        box_marker.color.r = 0.7
                        box_marker.color.g = 0.5
                        box_marker.color.b = 0.3
                    else:
                        box_marker.color.r = 0.5
                        box_marker.color.g = 0.5
                        box_marker.color.b = 0.5
                    
                    # Create box corners
                    p1 = Point(x=x1_world, y=y1_world, z=z_height)
                    p2 = Point(x=x2_world, y=y1_world, z=z_height)
                    p3 = Point(x=x2_world, y=y2_world, z=z_height)
                    p4 = Point(x=x1_world, y=y2_world, z=z_height)
                    
                    # Add points to form a closed loop
                    box_marker.points = [p1, p2, p3, p4, p1]
                    markers.markers.append(box_marker)
                    
                    # Add text marker with class name and confidence
                    text_marker = Marker()
                    text_marker.header.frame_id = frame_id
                    # Note: timestamp will be set by the caller
                    text_marker.ns = "debug_visualization"
                    text_marker.id = marker_id + 1000  # Offset to avoid ID collision
                    text_marker.type = Marker.TEXT_VIEW_FACING
                    text_marker.action = Marker.ADD
                    text_marker.scale.z = 0.05  # Text height
                    text_marker.color = box_marker.color
                    text_marker.color.a = 1.0
                    
                    # Position text above the box
                    text_marker.pose.position.x = (x1_world + x2_world) / 2
                    text_marker.pose.position.y = (y1_world + y2_world) / 2
                    text_marker.pose.position.z = z_height + 0.05  # Slightly above the grid
                    text_marker.pose.orientation.w = 1.0
                    
                    # Set text with class name and confidence (using underscores)
                    text_marker.text = f"{class_name}_{conf:.2f}"
                    markers.markers.append(text_marker)
                    
                    marker_id += 2  # Increment by 2 for box and text
                    
                except Exception as e:
                    if logger:
                        logger.error(f"Error creating marker for box: {e}")
                    continue
        
        # Add markers for detailed back regions if available
        if hasattr(detections, 'detailed_regions') and detections.detailed_regions:
            # Define region names for better visualization
            region_names = {
                "spine": "Spine",
                "left_lower": "Left_Lower_Back",
                "left_middle": "Left_Middle_Back",
                "left_upper": "Left_Upper_Back",
                "right_lower": "Right_Lower_Back",
                "right_middle": "Right_Middle_Back",
                "right_upper": "Right_Upper_Back"
            }
            
            # Create markers for each region
            for region_name, (points, color) in detections.detailed_regions.items():
                if len(points) == 0:
                    continue
                
                # Get region name
                display_name = region_names.get(region_name, region_name)
                
                # Create region boundary marker
                region_min = np.min(points, axis=0)
                region_max = np.max(points, axis=0)
                
                # Create box marker for region
                region_marker = Marker()
                region_marker.header.frame_id = frame_id
                # Note: timestamp will be set by the caller
                region_marker.ns = "debug_visualization"
                region_marker.id = marker_id
                region_marker.type = Marker.LINE_STRIP
                region_marker.action = Marker.ADD
                region_marker.scale.x = 0.003  # Thinner line width
                region_marker.color.a = 0.8
                region_marker.color.r = color[0]
                region_marker.color.g = color[1]
                region_marker.color.b = color[2]
                
                # Create box corners
                p1 = Point(x=region_min[0], y=region_min[1], z=z_height + 0.001)  # Slightly above grid
                p2 = Point(x=region_max[0], y=region_min[1], z=z_height + 0.001)
                p3 = Point(x=region_max[0], y=region_max[1], z=z_height + 0.001)
                p4 = Point(x=region_min[0], y=region_max[1], z=z_height + 0.001)
                
                # Add points to form a closed loop
                region_marker.points = [p1, p2, p3, p4, p1]
                markers.markers.append(region_marker)
                
                # Add text marker with region name
                text_marker = Marker()
                text_marker.header.frame_id = frame_id
                # Note: timestamp will be set by the caller
                text_marker.ns = "debug_visualization"
                text_marker.id = marker_id + 1000  # Offset to avoid ID collision
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.scale.z = 0.03  # Smaller text height
                text_marker.color.a = 1.0
                text_marker.color.r = color[0]
                text_marker.color.g = color[1]
                text_marker.color.b = color[2]
                
                # Position text above the region
                text_marker.pose.position.x = (region_min[0] + region_max[0]) / 2
                text_marker.pose.position.y = (region_min[1] + region_max[1]) / 2
                text_marker.pose.position.z = z_height + 0.03  # Slightly above the grid
                text_marker.pose.orientation.w = 1.0
                
                # Set text with region name and point count (using underscores)
                text_marker.text = f"{display_name}_{len(points)}pts"
                markers.markers.append(text_marker)
                
                marker_id += 2  # Increment by 2 for box and text
    
    return markers

def create_massage_region_markers(points_by_class, class_colors, robot_base_frame, detailed_regions=None, logger=None):
    """
    Create marker array for visualization of detected massage regions
    
    Args:
        points_by_class: Dictionary mapping class IDs to point arrays
        class_colors: Dictionary mapping class IDs to RGB color tuples
        robot_base_frame: Frame ID for the markers
        detailed_regions: Optional dictionary of detailed back regions
        logger: Optional ROS logger for debug messages
        
    Returns:
        MarkerArray with visualization markers
    """
    marker_array = MarkerArray()
    
    # Create markers for each class
    for class_id, points in points_by_class.items():
        # Skip empty point sets
        if points is None or len(points) == 0:
            continue
            
        # Skip torso (class 0) if we have detailed regions
        if class_id == 0 and detailed_regions and len(detailed_regions) > 0:
            if logger:
                logger.info("Skipping torso visualization in favor of detailed regions")
            continue
            
        # Get color for this class
        color = class_colors.get(class_id, (0.5, 0.5, 0.5))  # Default gray if class not found
        
        # Create point cloud marker
        marker = Marker()
        marker.header.frame_id = robot_base_frame
        # Note: timestamp will be set by the caller
        marker.ns = "body_regions"
        marker.id = class_id
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
            p.z = point[2]
            marker.points.append(p)
        
        marker_array.markers.append(marker)
        
        # Add text marker with class name
        text_marker = Marker()
        text_marker.header.frame_id = robot_base_frame
        # Note: timestamp will be set by the caller
        text_marker.ns = "body_regions"
        text_marker.id = class_id + 100  # Offset to avoid ID collision
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
            text_marker.pose.position.z = centroid[2] + 0.05  # Height offset
            text_marker.pose.orientation.w = 1.0
            
            # Use class ID as text if no name is provided
            text_marker.text = f"Class{class_id}"
            marker_array.markers.append(text_marker)
    
    # If we have detailed regions, add them to the visualization
    if detailed_regions:
        # Create markers for each detailed region
        region_id = 10  # Start IDs at 10 to avoid collision with original classes
        for region_name, (region_points, region_color) in detailed_regions.items():
            if len(region_points) == 0:
                continue
                
            # Create point cloud marker for this region
            marker = Marker()
            marker.header.frame_id = robot_base_frame
            # Note: timestamp will be set by the caller
            marker.ns = "body_regions"
            marker.id = region_id
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.scale.x = 0.02  # Point size
            marker.scale.y = 0.02
            marker.color.r = region_color[0]
            marker.color.g = region_color[1]
            marker.color.b = region_color[2]
            marker.color.a = 1.0
            
            # Add all points for this region
            for point in region_points:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = point[2]
                marker.points.append(p)
            
            marker_array.markers.append(marker)
            
            # Add text marker with region name
            text_marker = Marker()
            text_marker.header.frame_id = robot_base_frame
            # Note: timestamp will be set by the caller
            text_marker.ns = "body_regions"
            text_marker.id = 1000 + region_id  # Offset to avoid ID collision
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.scale.z = 0.05  # Text height
            text_marker.color.a = 1.0
            text_marker.color.r = region_color[0]
            text_marker.color.g = region_color[1]
            text_marker.color.b = region_color[2]
            
            # Calculate centroid of points for text placement
            centroid = np.mean(region_points, axis=0)
            text_marker.pose.position.x = centroid[0]
            text_marker.pose.position.y = centroid[1]
            text_marker.pose.position.z = centroid[2] + 0.05  # Height offset
            text_marker.pose.orientation.w = 1.0
            
            # Use the region name directly
            text_marker.text = region_name
            marker_array.markers.append(text_marker)
            
            region_id += 1
    
    return marker_array

def create_top_down_occupancy(points, crop_bounds, grid_resolution, save_grids=False, grid_dir=None, frame_counter=0, logger=None):
    """Create a 2D top-down occupancy grid from the point cloud"""
    try:
        # Get bounds for the grid
        x_min = crop_bounds["x_min"]
        x_max = crop_bounds["x_max"]
        y_min = crop_bounds["y_min"]
        y_max = crop_bounds["y_max"]
        
        # Calculate grid size
        grid_width = int(np.ceil((x_max - x_min) / grid_resolution))
        grid_height = int(np.ceil((y_max - y_min) / grid_resolution))
        
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
            x_idx = int((point[0] - x_min) / grid_resolution)
            y_idx = int((point[1] - y_min) / grid_resolution)
            
            # Check if the indices are within grid bounds
            if 0 <= x_idx < grid_width and 0 <= y_idx < grid_height:
                mask[x_idx, y_idx] = True
                # Store the first point we find for each cell (or could be replaced with criteria like lowest z)
                if representative_points[x_idx, y_idx] is None:
                    representative_points[x_idx, y_idx] = point
        
        if logger:
            logger.info(f"Created binary mask with dimensions {grid_width}x{grid_height}")
        
        # Create an image for YOLO with the same dimensions as our grid
        # IMPORTANT: YOLO expects images with (0,0) at the top-left
        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        if logger:
            logger.info(f"Mask shape: {mask.shape}, Grid image shape: {grid_image.shape}")
        
        # Fill the grid image
        # CRITICAL: We need to flip the y-axis when converting from our grid to the image
        # Our grid has (0,0) at bottom-left, but the image has (0,0) at top-left
        for i in range(grid_width):
            for j in range(grid_height):
                if mask[i, j]:
                    # Flip y-axis for the image (j → grid_height-1-j)
                    grid_image[grid_height-1-j, i] = [255, 255, 255]  # White for occupied cells
        
        # Save grid image if requested
        if save_grids and grid_dir:
            grid_image_path = os.path.join(grid_dir, f"grid_{frame_counter:04d}.png")
            try:
                import cv2
                cv2.imwrite(grid_image_path, grid_image)
                if logger:
                    logger.info(f"Saved grid image to {grid_image_path}")
            except Exception as e:
                if logger:
                    logger.error(f"Failed to save grid image: {str(e)}")
        
        return {
            "mask": mask,  # Binary mask for occupied/unoccupied cells
            "points": representative_points,  # Representative point for each cell
            "origin": (x_min, y_min),
            "image": grid_image,  # Image representation for YOLO
            "grid_dims": (grid_width, grid_height),  # Grid dimensions
            "resolution": grid_resolution  # Store resolution for later use
        }
    except Exception as e:
        if logger:
            logger.error(f"Error creating occupancy grid: {str(e)}")
        return None

def world_to_grid_coords(x_world, y_world, grid_data):
    """
    Convert world coordinates to grid indices
    
    Args:
        x_world: X coordinate in world frame
        y_world: Y coordinate in world frame
        grid_data: Dictionary with grid information
        
    Returns:
        Tuple of (i_idx, j_idx) grid indices
    """
    x_origin, y_origin = grid_data["origin"]
    grid_resolution = grid_data.get("resolution", 0.01)
    grid_width, grid_height = grid_data["mask"].shape
    
    # Convert to grid indices and ensure they're within bounds
    i_idx = max(0, min(grid_width - 1, int((x_world - x_origin) / grid_resolution)))
    j_idx = max(0, min(grid_height - 1, int((y_world - y_origin) / grid_resolution)))
    
    return i_idx, j_idx

def grid_to_world_coords(i_idx, j_idx, grid_data):
    """
    Convert grid indices to world coordinates
    
    Args:
        i_idx: Grid index in i direction (x)
        j_idx: Grid index in j direction (y)
        grid_data: Dictionary with grid information
        
    Returns:
        Tuple of (x_world, y_world) world coordinates
    """
    x_origin, y_origin = grid_data["origin"]
    grid_resolution = grid_data.get("resolution", 0.01)
    
    # Convert grid indices to world coordinates
    x_world = x_origin + i_idx * grid_resolution
    y_world = y_origin + j_idx * grid_resolution
    
    return x_world, y_world

def get_points_in_detection(points, grid_data, detection_box, structured_pattern=True, logger=None):
    """
    Extract points that fall within a detection bounding box, optionally in a structured pattern
    
    Args:
        points: Array of 3D points
        grid_data: Dictionary with grid information
        detection_box: YOLO detection box
        structured_pattern: If True, return points in alternating row pattern
        logger: Optional ROS logger for debug messages
        
    Returns:
        Numpy array of points within the detection box, optionally structured
    """
    try:
        # Convert the YOLO box to world coordinates
        x1_world, y1_world, x2_world, y2_world = convert_yolo_box_to_world_coords(
            detection_box, 
            grid_data, 
            logger=logger
        )
        
        # Log the world coordinates
        if logger:
            logger.info(f"World coordinates: [{x1_world:.2f}, {y1_world:.2f}, {x2_world:.2f}, {y2_world:.2f}]")
        
        # If we don't want structured pattern, just filter points directly
        if not structured_pattern:
            # Filter points that fall within the bounding box
            filtered_points = []
            for point in points:
                x, y, z = point
                if (x1_world <= x <= x2_world and 
                    y1_world <= y <= y2_world):
                    filtered_points.append(point)
            
            if logger:
                logger.info(f"Found {len(filtered_points)} points in detection region")
            
            return np.array(filtered_points) if filtered_points else None
        
        # For structured pattern, we need to use the grid
        else:
            # Extract grid data
            mask = grid_data["mask"]
            grid_width, grid_height = mask.shape
            x_origin, y_origin = grid_data["origin"]
            grid_resolution = grid_data.get("resolution", 0.01)
            
            # Convert world coordinates to grid indices
            x1_idx = max(0, min(grid_width - 1, int((x1_world - x_origin) / grid_resolution)))
            x2_idx = max(0, min(grid_width - 1, int((x2_world - x_origin) / grid_resolution)))
            y1_idx = max(0, min(grid_height - 1, int((y1_world - y_origin) / grid_resolution)))
            y2_idx = max(0, min(grid_height - 1, int((y2_world - y_origin) / grid_resolution)))
            
            # Create a grid for the detection region
            region_grid = np.zeros((grid_width, grid_height), dtype=bool)
            
            # Mark cells that contain points from the detection region
            for point in points:
                x, y, z = point
                if (x1_world <= x <= x2_world and y1_world <= y <= y2_world):
                    i = int((x - x_origin) / grid_resolution)
                    j = int((y - y_origin) / grid_resolution)
                    if 0 <= i < grid_width and 0 <= j < grid_height:
                        region_grid[i, j] = True
            
            # Create a mapping from grid cells to representative points
            cell_to_points = {}
            for point in points:
                x, y, z = point
                if (x1_world <= x <= x2_world and y1_world <= y <= y2_world):
                    i = int((x - x_origin) / grid_resolution)
                    j = int((y - y_origin) / grid_resolution)
                    if 0 <= i < grid_width and 0 <= j < grid_height:
                        if (i, j) not in cell_to_points:
                            cell_to_points[(i, j)] = []
                        cell_to_points[(i, j)].append(point)
            
            # Create structured points using alternating pattern
            structured_points = []
            
            # Process each row in the detection region with alternating direction
            for i in range(x1_idx, x2_idx + 1):
                # Determine direction for this row
                # Even rows (0, 2, 4...) go left to right, odd rows go right to left
                if i % 2 == 0:
                    # Left to right
                    j_range = range(y1_idx, y2_idx + 1)
                else:
                    # Right to left
                    j_range = range(y2_idx, y1_idx - 1, -1)
                
                # Process each cell in the row
                for j in j_range:
                    if region_grid[i, j]:
                        if (i, j) in cell_to_points:
                            # Use the average of all points in this cell
                            cell_points = cell_to_points[(i, j)]
                            avg_point = np.mean(cell_points, axis=0)
                            structured_points.append(avg_point)
            
            if logger:
                logger.info(f"Found {len(structured_points)} structured points in detection region")
            
            return np.array(structured_points) if structured_points else None
        
    except Exception as e:
        if logger:
            logger.error(f"Error extracting points in detection: {e}")
            import traceback
            logger.error(traceback.format_exc())
        return None

def create_detailed_back_regions(torso_points, legs_points, spine_width_fraction=5.0, back_regions_count=3, logger=None):
    """
    Divide the torso into detailed back regions:
    - Spine (center strip)
    - Left/Right Lower Back
    - Left/Right Middle Back
    - Left/Right Upper Back
    
    Args:
        torso_points: Points classified as torso
        legs_points: Points classified as legs
        spine_width_fraction: Spine width as fraction of torso width (default: 5.0)
        back_regions_count: Number of vertical regions to divide the back into (default: 3)
        logger: Optional ROS logger for debug messages
        
    Returns:
        Dictionary mapping region names to (points, color) tuples
    """
    try:
        if logger:
            logger.info(f"Creating detailed back regions from {len(torso_points)} torso points")
        
        # Define more distinct colors for detailed back regions
        back_region_colors = {
            "spine": (0.7, 0.7, 0.7),  # Gray
            "left_lower": (0.9, 0.1, 0.1),  # Bright red
            "left_middle": (0.1, 0.8, 0.1),  # Bright green
            "left_upper": (0.1, 0.1, 0.9),  # Bright blue
            "right_lower": (0.9, 0.6, 0.1),  # Orange
            "right_middle": (0.9, 0.1, 0.9),  # Magenta
            "right_upper": (0.1, 0.9, 0.9)  # Cyan
        }
        
        # Get the bounding box of the torso
        torso_min = np.min(torso_points, axis=0)
        torso_max = np.max(torso_points, axis=0)
        
        # Get the bounding box of the legs
        legs_min = np.min(legs_points, axis=0)
        legs_max = np.max(legs_points, axis=0)
        
        # Determine the direction from legs to torso (this is the "up" direction)
        # We assume the legs are below the torso in the robot's coordinate frame
        legs_center = (legs_min + legs_max) / 2
        torso_center = (torso_min + torso_max) / 2
        
        # Calculate the spine width (1/5 of the torso width)
        torso_width = torso_max[1] - torso_min[1]  # Y-axis is typically width in robot frame
        spine_width = torso_width / spine_width_fraction
        
        # Calculate the spine center line
        spine_center_y = (torso_min[1] + torso_max[1]) / 2.0
        spine_min_y = spine_center_y - spine_width / 2.0
        spine_max_y = spine_center_y + spine_width / 2.0
        
        # Divide the torso height into three equal parts (lower, middle, upper)
        torso_height = torso_max[0] - torso_min[0]  # X-axis is typically height in robot frame
        section_height = torso_height / back_regions_count
        
        # Define the height boundaries for each section
        section_boundaries = []
        for i in range(back_regions_count + 1):
            section_boundaries.append(torso_min[0] + i * section_height)
        
        if logger:
            logger.info(f"Torso bounds: X=[{torso_min[0]:.2f}, {torso_max[0]:.2f}], Y=[{torso_min[1]:.2f}, {torso_max[1]:.2f}]")
            logger.info(f"Spine region: Y=[{spine_min_y:.2f}, {spine_max_y:.2f}]")
            logger.info(f"Height sections: {section_boundaries}")
        
        # Initialize region points dictionary
        region_points = {
            "spine": [],
            "left_lower": [],
            "left_middle": [],
            "left_upper": [],
            "right_lower": [],
            "right_middle": [],
            "right_upper": []
        }
        
        # Assign each torso point to the appropriate region
        for point in torso_points:
            x, y, z = point
            
            # Check if point is in the spine region
            if spine_min_y <= y <= spine_max_y:
                region_points["spine"].append(point)
            # Left side regions (y < spine_min_y)
            elif y < spine_min_y:
                # Determine vertical region (lower, middle, upper)
                if section_boundaries[0] <= x < section_boundaries[1]:
                    region_points["left_lower"].append(point)
                elif section_boundaries[1] <= x < section_boundaries[2]:
                    region_points["left_middle"].append(point)
                elif section_boundaries[2] <= x <= section_boundaries[3]:
                    region_points["left_upper"].append(point)
            # Right side regions (y > spine_max_y)
            elif y > spine_max_y:
                # Determine vertical region (lower, middle, upper)
                if section_boundaries[0] <= x < section_boundaries[1]:
                    region_points["right_lower"].append(point)
                elif section_boundaries[1] <= x < section_boundaries[2]:
                    region_points["right_middle"].append(point)
                elif section_boundaries[2] <= x <= section_boundaries[3]:
                    region_points["right_upper"].append(point)
        
        # Create the dictionary of regions with points and colors
        regions = {}
        for name, points in region_points.items():
            if points:  # Only include regions with points
                regions[name] = (np.array(points), back_region_colors[name])
                if logger:
                    logger.info(f"Region '{name}' has {len(points)} points with color {back_region_colors[name]}")
        
        return regions
        
    except Exception as e:
        if logger:
            logger.error(f"Error creating detailed back regions: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        return {}

def get_best_detections(yolo_result, logger=None):
    """Get the highest confidence detection for each class"""
    boxes_by_class = {}
    for box in yolo_result.boxes:
        class_instance_id = int(box.cls.item())
        conf = float(box.conf.item())
        
        # Only consider head (1), torso (0), and legs (6)
        if class_instance_id not in [0, 1, 6]:
            continue
        
        # Keep the highest confidence detection for each class
        if class_instance_id not in boxes_by_class or conf > boxes_by_class[class_instance_id][1]:
            boxes_by_class[class_instance_id] = (box, conf)
            
    return boxes_by_class

def run_yolo_inference(model, grid_image, yolo_low_conf_threshold=0.1, cv_bridge=None, 
                       yolo_input_pub=None, yolo_output_pub=None, robot_base_frame=None, 
                       logger=None):
    """Run YOLO inference directly on the grid image numpy array"""
    if model is None:
        if logger:
            logger.warn("YOLO model not available, skipping inference")
        return None
    
    try:
        # Get original image dimensions
        original_height, original_width = grid_image.shape[:2]
        if logger:
            logger.info(f"Original grid image dimensions: {original_width}x{original_height}")
        
        # Publish the input image for visualization if publishers are provided
        if yolo_input_pub and cv_bridge and robot_base_frame:
            try:
                img_msg = cv_bridge.cv2_to_imgmsg(grid_image, encoding="rgb8")
                img_msg.header.frame_id = robot_base_frame
                yolo_input_pub.publish(img_msg)
                if logger:
                    logger.info("Published YOLO input image")
            except Exception as e:
                if logger:
                    logger.error(f"Error publishing input image: {str(e)}")
        
        # Run inference directly on the numpy array
        if logger:
            logger.info(f"Running YOLO inference on grid image array")
        results = model(grid_image)
        
        # Check if we got any detections
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Store original image dimensions in the result for later use
            results[0].orig_shape = grid_image.shape[:2]  # (height, width)
            
            if logger:
                logger.info(f"YOLO detected {len(results[0].boxes)} objects")
                for i, box in enumerate(results[0].boxes):
                    class_instance_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    xyxy = box.xyxy[0].cpu().numpy()
                    logger.info(f"Detection {i}: class={class_instance_id}, conf={conf:.2f}, box={xyxy}")
            
            # Create output image with detection boxes
            output_img = results[0].plot()
            
            # Publish the output image if publishers are provided
            if yolo_output_pub and cv_bridge and robot_base_frame:
                try:
                    img_msg = cv_bridge.cv2_to_imgmsg(output_img, encoding="rgb8")
                    img_msg.header.frame_id = robot_base_frame
                    yolo_output_pub.publish(img_msg)
                    if logger:
                        logger.info("Published YOLO output image with detections")
                except Exception as e:
                    if logger:
                        logger.error(f"Error publishing output image: {str(e)}")
            
            return results[0]
        else:
            if logger:
                logger.warn("YOLO detected no objects")
            
            # Try with lower confidence threshold
            if logger:
                logger.info(f"Trying YOLO with increased confidence threshold ({yolo_low_conf_threshold})")
            results = model(grid_image, conf=yolo_low_conf_threshold)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Store original image dimensions in the result for later use
                results[0].orig_shape = grid_image.shape[:2]  # (height, width)
                
                if logger:
                    logger.info(f"YOLO detected {len(results[0].boxes)} objects with lower threshold")
                    for i, box in enumerate(results[0].boxes):
                        class_instance_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        xyxy = box.xyxy[0].cpu().numpy()
                        logger.info(f"Detection {i}: class={class_instance_id}, conf={conf:.2f}, box={xyxy}")
                
                # Create output image with detection boxes
                output_img = results[0].plot()
                
                # Publish the output image if publishers are provided
                if yolo_output_pub and cv_bridge and robot_base_frame:
                    try:
                        img_msg = cv_bridge.cv2_to_imgmsg(output_img, encoding="rgb8")
                        img_msg.header.frame_id = robot_base_frame
                        yolo_output_pub.publish(img_msg)
                        if logger:
                            logger.info("Published YOLO output image with detections (lower threshold)")
                    except Exception as e:
                        if logger:
                            logger.error(f"Error publishing output image: {str(e)}")
                
                return results[0]
            
            return None
    except Exception as e:
        if logger:
            logger.error(f"Error during YOLO inference: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        return None

def create_numbered_point_markers(points, frame_id, marker_namespace="numbered_points", z_offset=0.01, 
                                 stride=5, section_based=False, logger=None, 
                                 massage_gun_tip_transform=None, visualize_path=False, region_color=None):
    """
    Create markers to visualize numbered points showing the order of structured points
    
    Args:
        points: Array of 3D points
        frame_id: Frame ID for the markers
        marker_namespace: Namespace for the markers
        z_offset: Height offset for the text markers
        stride: Show every Nth point (default: 5)
        section_based: If True, reset numbering for each section (row) of points
        logger: Optional ROS logger for debug messages
        massage_gun_tip_transform: Optional [x,y,z] translation to apply to points for massage gun tip
        visualize_path: If True, create a path visualization for massage gun tip points
        region_color: Optional color to use for this region's visualization
        
    Returns:
        MarkerArray with numbered point visualization
    """
    try:
        if points is None or len(points) == 0:
            if logger:
                logger.warn("No points to visualize")
            return MarkerArray()
            
        markers = MarkerArray()
        
        # Define region-specific colors if not provided
        if region_color is None:
            # Extract region name from namespace if possible
            region_name = marker_namespace.split('_')[-1] if '_' in marker_namespace else ""
            
            # Generate a color based on the region name or use a default
            if region_name == "upper_back":
                region_color = (0.2, 0.8, 0.2)  # Green
            elif region_name == "mid_back":
                region_color = (0.2, 0.2, 0.8)  # Blue
            elif region_name == "lower_back":
                region_color = (0.8, 0.2, 0.2)  # Red
            elif region_name == "left_back":
                region_color = (0.8, 0.8, 0.2)  # Yellow
            elif region_name == "right_back":
                region_color = (0.8, 0.2, 0.8)  # Purple
            elif region_name == "spine":
                region_color = (0.2, 0.8, 0.8)  # Cyan
            else:
                # Generate a pseudo-random color based on the namespace string
                import hashlib
                hash_val = int(hashlib.md5(marker_namespace.encode()).hexdigest(), 16)
                r = ((hash_val & 0xFF0000) >> 16) / 255.0
                g = ((hash_val & 0x00FF00) >> 8) / 255.0
                b = (hash_val & 0x0000FF) / 255.0
                region_color = (r, g, b)
        
        # Skip creating markers for original points - only show translated points
        
        # For section-based numbering, we need to identify rows
        if section_based:
            # Group points by rows (based on x-coordinate)
            # First, sort points by x-coordinate to identify rows
            points_with_indices = list(enumerate(points))
            points_with_indices.sort(key=lambda p: p[1][0])  # Sort by x-coordinate
            
            # Identify row boundaries by looking for significant changes in x-coordinate
            rows = []
            current_row = [points_with_indices[0]]
            current_x = points_with_indices[0][1][0]
            
            for i, (idx, point) in enumerate(points_with_indices[1:], 1):
                if abs(point[0] - current_x) > 0.02:  # 2cm threshold for new row
                    # Found a new row
                    rows.append(current_row)
                    current_row = [(idx, point)]
                    current_x = point[0]
                else:
                    current_row.append((idx, point))
            
            # Add the last row
            if current_row:
                rows.append(current_row)
            
            if logger:
                logger.info(f"Identified {len(rows)} rows in the point set")
            
            # Create line markers for each row with translated points
            for row_idx, row in enumerate(rows):
                # Sort points within this row by y-coordinate
                # For even rows, sort left to right (increasing y)
                # For odd rows, sort right to left (decreasing y)
                if row_idx % 2 == 0:
                    row.sort(key=lambda p: p[1][1])  # Sort by y-coordinate (increasing)
                else:
                    row.sort(key=lambda p: -p[1][1])  # Sort by y-coordinate (decreasing)
                
                # Apply stride to get the points we'll use
                strided_row = [row[i] for i in range(0, len(row), stride)]
                if len(row) > 0 and (len(row) - 1) % stride != 0:
                    strided_row.append(row[-1])  # Add the last point if not already included
                
                # Skip if no points after stride
                if not strided_row:
                    continue
                
                # Create translated points for this row
                translated_points = []
                for _, point in strided_row:
                    if massage_gun_tip_transform:
                        translated_point = [
                            point[0] + massage_gun_tip_transform[0],
                            point[1] + massage_gun_tip_transform[1],
                            point[2] + massage_gun_tip_transform[2]
                        ]
                        translated_points.append(translated_point)
                    else:
                        translated_points.append(point)
                
                # Create a marker for the translated points
                points_marker = Marker()
                points_marker.header.frame_id = frame_id
                points_marker.ns = f"{marker_namespace}_row_{row_idx}_translated_points"
                points_marker.id = row_idx + 1000
                points_marker.type = Marker.POINTS
                points_marker.action = Marker.ADD
                points_marker.scale.x = 0.015  # Larger point size
                points_marker.scale.y = 0.015
                points_marker.color.a = 1.0
                points_marker.color.r = region_color[0]
                points_marker.color.g = region_color[1]
                points_marker.color.b = region_color[2]
                
                # Add all translated points
                for point in translated_points:
                    p = Point()
                    p.x = point[0]
                    p.y = point[1]
                    p.z = point[2]
                    points_marker.points.append(p)
                
                markers.markers.append(points_marker)
                
                # Create line marker for this row with translated points
                line_marker = Marker()
                line_marker.header.frame_id = frame_id
                line_marker.ns = f"{marker_namespace}_row_{row_idx}_translated_path"
                line_marker.id = row_idx + 2000
                line_marker.type = Marker.LINE_STRIP
                line_marker.action = Marker.ADD
                line_marker.scale.x = 0.008  # Thicker line width for better visibility
                
                # Use region color with slight variation for rows
                line_marker.color.a = 1.0
                line_marker.color.r = min(1.0, region_color[0] * (1.0 + 0.2 * (-1 if row_idx % 2 == 0 else 1)))
                line_marker.color.g = min(1.0, region_color[1] * (1.0 + 0.2 * (-1 if row_idx % 2 == 0 else 1)))
                line_marker.color.b = min(1.0, region_color[2] * (1.0 + 0.2 * (-1 if row_idx % 2 == 0 else 1)))
                
                # Add translated points to form a path for this row
                for point in translated_points:
                    p = Point()
                    p.x = point[0]
                    p.y = point[1]
                    p.z = point[2]
                    line_marker.points.append(p)
                
                markers.markers.append(line_marker)
                
                # Create text markers for translated points in this row
                for i, point in enumerate(translated_points):
                    text_marker = Marker()
                    text_marker.header.frame_id = frame_id
                    text_marker.ns = f"{marker_namespace}_row_{row_idx}_text"
                    text_marker.id = 3000 + row_idx * 100 + i
                    text_marker.type = Marker.TEXT_VIEW_FACING
                    text_marker.action = Marker.ADD
                    text_marker.scale.z = 0.02  # Text height
                    text_marker.color.a = 1.0
                    text_marker.color.r = 1.0
                    text_marker.color.g = 1.0
                    text_marker.color.b = 1.0
                    
                    # Position text above the translated point
                    text_marker.pose.position.x = point[0]
                    text_marker.pose.position.y = point[1]
                    text_marker.pose.position.z = point[2] + z_offset
                    text_marker.pose.orientation.w = 1.0
                    
                    # Set text with row and point index
                    text_marker.text = f"{row_idx}:{i}"
                    markers.markers.append(text_marker)
            
            # Create connections between rows (vertical connections)
            if len(rows) > 1 and visualize_path:
                for i in range(len(rows) - 1):
                    # Get the last point of the current row and the first point of the next row
                    current_row = rows[i]
                    next_row = rows[i+1]
                    
                    if not current_row or not next_row:
                        continue
                    
                    # Apply stride to get the points we'll use
                    strided_current_row = [current_row[j] for j in range(0, len(current_row), stride)]
                    if len(current_row) > 0 and (len(current_row) - 1) % stride != 0:
                        strided_current_row.append(current_row[-1])
                        
                    strided_next_row = [next_row[j] for j in range(0, len(next_row), stride)]
                    if len(next_row) > 0 and (len(next_row) - 1) % stride != 0:
                        strided_next_row.append(next_row[-1])
                    
                    if not strided_current_row or not strided_next_row:
                        continue
                    
                    # Sort rows by y-coordinate for consistent connections
                    if i % 2 == 0:  # Current row is left-to-right
                        _, last_point = strided_current_row[-1]
                        _, first_point = strided_next_row[0]
                    else:  # Current row is right-to-left
                        _, last_point = strided_current_row[-1]
                        _, first_point = strided_next_row[-1]
                    
                    # Create translated points
                    if massage_gun_tip_transform:
                        last_translated = [
                            last_point[0] + massage_gun_tip_transform[0],
                            last_point[1] + massage_gun_tip_transform[1],
                            last_point[2] + massage_gun_tip_transform[2]
                        ]
                        first_translated = [
                            first_point[0] + massage_gun_tip_transform[0],
                            first_point[1] + massage_gun_tip_transform[1],
                            first_point[2] + massage_gun_tip_transform[2]
                        ]
                    else:
                        last_translated = last_point
                        first_translated = first_point
                    
                    # Create a connector line marker
                    connector_marker = Marker()
                    connector_marker.header.frame_id = frame_id
                    connector_marker.ns = f"{marker_namespace}_row_connector_{i}"
                    connector_marker.id = 4000 + i
                    connector_marker.type = Marker.LINE_LIST
                    connector_marker.action = Marker.ADD
                    connector_marker.scale.x = 0.008  # Thicker line width
                    connector_marker.color.a = 1.0
                    connector_marker.color.r = region_color[0]
                    connector_marker.color.g = region_color[1]
                    connector_marker.color.b = region_color[2]
                    
                    # Add the two points to form a line
                    p1 = Point()
                    p1.x = last_translated[0]
                    p1.y = last_translated[1]
                    p1.z = last_translated[2]
                    connector_marker.points.append(p1)
                    
                    p2 = Point()
                    p2.x = first_translated[0]
                    p2.y = first_translated[1]
                    p2.z = first_translated[2]
                    connector_marker.points.append(p2)
                    
                    markers.markers.append(connector_marker)
        else:
            # Non-section based numbering (global)
            # Only show translated points
            if massage_gun_tip_transform and visualize_path:
                # Create a marker for the massage gun tip path
                tip_path_marker = Marker()
                tip_path_marker.header.frame_id = frame_id
                tip_path_marker.ns = f"{marker_namespace}_tip_path"
                tip_path_marker.id = 2000
                tip_path_marker.type = Marker.LINE_STRIP
                tip_path_marker.action = Marker.ADD
                tip_path_marker.scale.x = 0.008  # Thicker line width
                tip_path_marker.color.a = 1.0
                tip_path_marker.color.r = region_color[0]
                tip_path_marker.color.g = region_color[1]
                tip_path_marker.color.b = region_color[2]
                
                # Create a marker for the massage gun tip points
                tip_points_marker = Marker()
                tip_points_marker.header.frame_id = frame_id
                tip_points_marker.ns = f"{marker_namespace}_tip_points"
                tip_points_marker.id = 3000
                tip_points_marker.type = Marker.POINTS
                tip_points_marker.action = Marker.ADD
                tip_points_marker.scale.x = 0.015  # Larger point size
                tip_points_marker.scale.y = 0.015
                tip_points_marker.color.a = 1.0
                tip_points_marker.color.r = region_color[0]
                tip_points_marker.color.g = region_color[1]
                tip_points_marker.color.b = region_color[2]
                
                # Add points to the path with the stride
                for i, point in enumerate(points):
                    if i % stride == 0 or i == len(points) - 1:  # Apply stride
                        p = Point()
                        p.x = point[0] + massage_gun_tip_transform[0]
                        p.y = point[1] + massage_gun_tip_transform[1]
                        p.z = point[2] + massage_gun_tip_transform[2]
                        tip_path_marker.points.append(p)
                        tip_points_marker.points.append(p)
                
                # Only add markers if they have points
                if len(tip_path_marker.points) > 0:
                    markers.markers.append(tip_path_marker)
                    markers.markers.append(tip_points_marker)
        
        if logger:
            logger.info(f"Created {len(markers.markers)} markers for {len(points)} numbered points (stride={stride}, section_based={section_based})")
        
        return markers
        
    except Exception as e:
        if logger:
            logger.error(f"Error creating numbered point markers: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        return MarkerArray()

def process_detections(best_detections, cropped_points, grid_data, class_colors, robot_base_frame, 
                       detection_publisher=None, marker_publisher=None, crop_bounds=None, logger=None,
                       visualize_point_order=False, point_stride=5, 
                       massage_gun_tip_transform=None, visualize_path=False):
    """Process YOLO detections and create visualization markers"""
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
                logger.info("Published detection markers")
        
        # Create and publish numbered point markers if requested
        if visualize_point_order and marker_publisher and detailed_regions:
            # Only visualize the detailed back regions (not the class-based regions)
            for region_name, (region_points, region_color) in detailed_regions.items():
                # Skip the spine region for motion planning
                if region_name == "spine":
                    if logger:
                        logger.info(f"Skipping motion planning for spine region")
                    continue
                
                # Create a unique namespace for each region
                namespace = f"numbered_points_region_{region_name}"
                
                if logger:
                    logger.info(f"Creating markers for region {region_name} with color {region_color}")
                
                # Create numbered point markers with section-based numbering
                numbered_markers = create_numbered_point_markers(
                    region_points,
                    robot_base_frame,
                    marker_namespace=namespace,
                    stride=point_stride,
                    section_based=True,  # Enable section-based numbering
                    logger=logger,
                    massage_gun_tip_transform=massage_gun_tip_transform,
                    visualize_path=visualize_path,
                    region_color=region_color
                )
                
                # Publish numbered point markers
                marker_publisher.publish(numbered_markers)
                if logger:
                    logger.info(f"Published section-based numbered point markers for region {region_name} (stride={point_stride})")
        
        # Return the detected points by class
        return points_by_class, detailed_regions
    
    return {}, {}

def publish_image(image, publisher, robot_base_frame, cv_bridge, timestamp=None, log_message=None, logger=None):
    """Helper method to publish images with proper headers"""
    try:
        img_msg = cv_bridge.cv2_to_imgmsg(image, encoding="rgb8")
        if timestamp:
            img_msg.header.stamp = timestamp
        img_msg.header.frame_id = robot_base_frame
        publisher.publish(img_msg)
        if log_message and logger:
            logger.info(log_message)
        return True
    except Exception as e:
        if logger:
            logger.error(f"Error publishing image: {str(e)}")
        return False 