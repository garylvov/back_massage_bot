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
                x = x_origin + (i + 0.5) * grid_resolution
                y = y_origin + (j + 0.5) * grid_resolution
                
                point = Point()
                point.x = x
                point.y = y
                point.z = z_height
                grid_marker.points.append(point)
    
    markers.markers.append(grid_marker)
    
    # Create an outline of the grid area
    outline = Marker()
    outline.header.frame_id = frame_id
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
    markers.markers.append(outline)
    
    # Check if we have a custom object with both detections and detailed regions
    detailed_regions = None
    if hasattr(detections, 'detections') and hasattr(detections, 'detailed_regions'):
        detailed_regions = detections.detailed_regions
        detections = detections.detections
        if logger:
            logger.info(f"Found custom detection object with detailed regions")
    
    # Add detection box markers if we have detections
    if detections is not None:
        # Define default colors for visualization
        colors = [
            (1.0, 0.0, 0.0),  # Red
            (0.0, 1.0, 0.0),  # Green
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 1.0, 0.0),  # Yellow
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0),  # Cyan
            (0.7, 0.5, 0.3),  # Brown
            (0.5, 0.5, 0.5)   # Gray
        ]
        
        # Check if we have a YOLO result object or a filtered dictionary
        if hasattr(detections, 'boxes'):
            # It's a YOLO result object
            boxes = detections.boxes
            for i, box in enumerate(boxes):
                # Get box coordinates in world frame
                x1_world, y1_world, x2_world, y2_world = convert_yolo_box_to_world_coords(box, grid_data, logger)
                
                # Log the detection box coordinates for debugging
                if logger:
                    class_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    logger.info(f"Debug marker for detection class={class_id}, conf={conf:.2f}, world=[{x1_world:.2f},{y1_world:.2f},{x2_world:.2f},{y2_world:.2f}]")
                
                # Create box marker
                box_marker = Marker()
                box_marker.header.frame_id = frame_id
                # Note: timestamp will be set by the caller
                box_marker.ns = "debug_visualization"
                box_marker.id = 10 + i  # Use index to avoid ID collisions
                box_marker.type = Marker.LINE_STRIP
                box_marker.action = Marker.ADD
                box_marker.scale.x = 0.005  # Line width
                box_marker.color.a = 1.0
                
                # Get color from the box if available, otherwise use a default color
                if hasattr(box, 'color'):
                    color = box.color
                else:
                    # Use a default color scheme based on class ID
                    class_id = int(box.cls.item())
                    color_idx = class_id % len(colors)
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
                markers.markers.append(box_marker)
                
                # Add text marker with class name or ID
                text_marker = Marker()
                text_marker.header.frame_id = frame_id
                # Note: timestamp will be set by the caller
                text_marker.ns = "debug_visualization"
                text_marker.id = 100 + i  # Offset to avoid ID collision
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
                
                # Get class name if available
                if hasattr(box, 'class_name'):
                    text_marker.text = box.class_name
                else:
                    class_id = int(box.cls.item())
                    text_marker.text = f"Class{class_id}"
                
                markers.markers.append(text_marker)
        else:
            # It's a filtered dictionary {class_id: (box, conf)}
            for i, (class_id, (box, conf)) in enumerate(detections.items()):
                # Get box coordinates in world frame
                x1_world, y1_world, x2_world, y2_world = convert_yolo_box_to_world_coords(box, grid_data, logger)
                
                # Log the detection box coordinates for debugging
                if logger:
                    logger.info(f"Debug marker for detection class={class_id}, conf={conf:.2f}, world=[{x1_world:.2f},{y1_world:.2f},{x2_world:.2f},{y2_world:.2f}]")
                
                # Create box marker
                box_marker = Marker()
                box_marker.header.frame_id = frame_id
                # Note: timestamp will be set by the caller
                box_marker.ns = "debug_visualization"
                box_marker.id = 10 + i  # Use index to avoid ID collisions
                box_marker.type = Marker.LINE_STRIP
                box_marker.action = Marker.ADD
                box_marker.scale.x = 0.005  # Line width
                box_marker.color.a = 1.0
                
                # Use a default color scheme based on class ID
                color_idx = class_id % len(colors)
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
                markers.markers.append(box_marker)
                
                # Add text marker with class name or ID
                text_marker = Marker()
                text_marker.header.frame_id = frame_id
                # Note: timestamp will be set by the caller
                text_marker.ns = "debug_visualization"
                text_marker.id = 100 + i  # Offset to avoid ID collision
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
                
                # Use class ID as text
                text_marker.text = f"Class{class_id}"
                
                markers.markers.append(text_marker)
    
    # If we have detailed regions, add them to the debug visualization
    if detailed_regions:
        # Create markers for each detailed region
        region_id = 200  # Start IDs at 200 to avoid collision with other markers
        for region_name, (region_points, region_color) in detailed_regions.items():
            if len(region_points) == 0:
                continue
                
            # Create a box marker for this region
            region_min = np.min(region_points, axis=0)
            region_max = np.max(region_points, axis=0)
            
            # Create box marker
            box_marker = Marker()
            box_marker.header.frame_id = frame_id
            # Note: timestamp will be set by the caller
            box_marker.ns = "debug_visualization"
            box_marker.id = region_id
            box_marker.type = Marker.LINE_STRIP
            box_marker.action = Marker.ADD
            box_marker.scale.x = 0.003  # Line width (thinner than main boxes)
            box_marker.color.a = 1.0
            box_marker.color.r = region_color[0]
            box_marker.color.g = region_color[1]
            box_marker.color.b = region_color[2]
            box_marker.pose.orientation.w = 1.0
            
            # Create the box outline
            z = z_height + 0.002  # Slightly above the other boxes
            corners = [
                (region_min[0], region_min[1], z),
                (region_max[0], region_min[1], z),
                (region_max[0], region_max[1], z),
                (region_min[0], region_max[1], z),
                (region_min[0], region_min[1], z)  # Close the loop
            ]
            
            box_marker.points = [Point(x=x, y=y, z=z) for x, y, z in corners]
            markers.markers.append(box_marker)
            
            # Add text marker with region name
            text_marker = Marker()
            text_marker.header.frame_id = frame_id
            # Note: timestamp will be set by the caller
            text_marker.ns = "debug_visualization"
            text_marker.id = region_id + 100  # Offset to avoid ID collision
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.scale.z = 0.03  # Text height (smaller than main labels)
            text_marker.color.a = 1.0
            text_marker.color.r = region_color[0]
            text_marker.color.g = region_color[1]
            text_marker.color.b = region_color[2]
            
            # Position text in the center of the region
            text_marker.pose.position.x = (region_min[0] + region_max[0]) / 2
            text_marker.pose.position.y = (region_min[1] + region_max[1]) / 2
            text_marker.pose.position.z = z + 0.03  # Above the box
            text_marker.pose.orientation.w = 1.0
            
            # Use the region name directly
            text_marker.text = region_name
            markers.markers.append(text_marker)
            
            region_id += 1
    
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

def get_points_in_detection(points, grid_data, detection_box, logger=None):
    """Extract points that fall within a detection bounding box"""
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
        
    except Exception as e:
        if logger:
            logger.error(f"Error extracting points in detection: {e}")
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
        
        # Define colors for detailed back regions
        back_region_colors = {
            "spine": (0.7, 0.7, 0.7),  # Gray
            "left_lower": (0.8, 0.2, 0.2),  # Dark red
            "left_middle": (1.0, 0.4, 0.4),  # Medium red
            "left_upper": (1.0, 0.6, 0.6),  # Light red
            "right_lower": (0.2, 0.2, 0.8),  # Dark blue
            "right_middle": (0.4, 0.4, 1.0),  # Medium blue
            "right_upper": (0.6, 0.6, 1.0)  # Light blue
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
                if section_boundaries[0] <= x < section_boundaries[1]:
                    region_points["left_lower"].append(point)
                elif section_boundaries[1] <= x < section_boundaries[2]:
                    region_points["left_middle"].append(point)
                elif section_boundaries[2] <= x <= section_boundaries[3]:
                    region_points["left_upper"].append(point)
            # Right side regions (y > spine_max_y)
            elif y > spine_max_y:
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
        
        # Log the number of points in each region
        if logger:
            for name, (points, _) in regions.items():
                logger.info(f"Region '{name}' has {len(points)} points")
        
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

def process_detections(best_detections, cropped_points, grid_data, class_colors, robot_base_frame, 
                       detection_publisher=None, marker_publisher=None, crop_bounds=None, logger=None):
    """Process YOLO detections and create visualization markers"""
    # Extract points for each detection
    points_by_class = {}
    for class_instance_id, (box, conf) in best_detections.items():
        detection_points = get_points_in_detection(
            cropped_points, 
            grid_data, 
            box,
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
            detailed_regions,
            logger=logger
        )
            
        if detection_markers.markers and detection_publisher:
            detection_publisher.publish(detection_markers)
            if logger:
                logger.info(f"Published detection markers with {len(detection_markers.markers)} regions")
            
            # Create debug visualization showing grid and detection boxes
            # For the debug visualization, we'll create a custom object that has
            # both the detections and detailed regions
            
            class DetectionResult:
                def __init__(self):
                    self.boxes = []
                    self.detailed_regions = None
            
            # Create a custom result object
            result = DetectionResult()
            
            # Add boxes from best_detections
            for class_id, (box, conf) in best_detections.items():
                result.boxes.append(box)
            
            # Add detailed regions
            result.detailed_regions = detailed_regions
            
            if marker_publisher:
                debug_markers = create_yolo_markers(
                    grid_data, 
                    result,  # Pass our custom result object
                    robot_base_frame, 
                    grid_data.get("resolution", 0.01), 
                    crop_bounds["z_min"] if crop_bounds else 0.0,
                    logger=logger
                )
                
                marker_publisher.publish(debug_markers)
                if logger:
                    logger.info(f"Published visualization markers")
        else:
            if logger:
                logger.warn("No detection markers created")
    else:
        if logger:
            logger.warn("No points found in detection regions")
            
        # Create empty debug visualization if no detections
        if marker_publisher and crop_bounds:
            debug_markers = create_yolo_markers(
                grid_data, 
                None,  # No detections
                robot_base_frame, 
                grid_data.get("resolution", 0.01), 
                crop_bounds["z_min"],
                logger=logger
            )
            marker_publisher.publish(debug_markers)
            if logger:
                logger.info("Published empty visualization markers")
    
    return points_by_class, detailed_regions

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