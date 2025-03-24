import numpy as np
import cv2
from typing import List
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Rectangle
from scipy import ndimage


# Define YOLO class mapping with legs grouped together
YOLO_CLASS_MAPPING = [
    "Torso",
    "Head",
    "left_upper_arm",
    "right_upper_arm",
    "left_lower_arm",
    "right_lower_arm",
    "legs"  # All legs (left and right) grouped together
]

# Helper function to visualize results of multiple command chains
def visualize_command_results(original_image, command_chains, bboxes=None, title=None):
    n_chains = len(command_chains)
    rows = max(1, int(np.ceil(n_chains / 3)))
    cols = min(3, n_chains)
    
    plt.figure(figsize=(5*cols, 4*rows))
    if title:
        plt.suptitle(title, fontsize=16)
    
    for i, chain in enumerate(command_chains):
        ax = plt.subplot(rows, cols, i+1)
        result = chain.execute(original_image)
        plt.imshow(result, cmap='gray')
        plt.title(chain.get_name())
        plt.colorbar()
        
        # Add bounding boxes if provided
        if bboxes:
            draw_bounding_boxes(ax, bboxes)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    plt.show()
        
    
def sample_from_params(params: dict[str, tuple[float, float]],
                    strategy: str = "uniform") -> dict[str, float]:
    sampled_values = {}
    
    for key, bounds in params.items():
        # Sort bounds to ensure lower < upper
        lower, upper = sorted(bounds)
        
        if strategy == "uniform":
            sampled_values[key] = np.random.uniform(lower, upper)
        elif strategy == "normal":
            mean = (lower + upper) / 2
            std_dev = (upper - lower) / 6
            sampled_values[key] = np.random.normal(mean, std_dev)
            
    return sampled_values

def generate_empty_grid(
    width_meters: float = 1.5,
    height_meters: float = 2,
    resolution: float = 0.01,
) -> np.ndarray:
    """Generate an empty grid."""
    width_pixels = int(width_meters / resolution)
    height_pixels = int(height_meters / resolution)
    return np.zeros((height_pixels, width_pixels), dtype=np.uint8)

def create_filled_curve_mask(
    shape: tuple[int, int],  # (height, width) in pixels
    points: list[tuple[float, float]],  # List of (x, y) points in meters
    grid_resolution: float = 0.01,
    fill_value: int = 255,  # Value for the filled area
    curves_enabled: bool = False,  # Default to straight lines for stability
    curve_tension: float = 0.025,  # Much tighter curves (significantly reduced)
) -> np.ndarray:
    """
    Create a binary mask with a filled curve that passes through ALL the given points.
    
    Args:
        shape: Tuple of (height, width) for the mask in pixels
        points: List of points (x, y) in meters that the curve passes through
        grid_resolution: Grid resolution in meters/cell
        fill_value: Value to use for the filled area (typically 255 for white)
        curves_enabled: Whether to use curved edges (True) or straight lines (False)
        curve_tension: Controls how tightly the curve follows the points (lower = tighter)
        
    Returns:
        Binary mask where the enclosed region is filled with fill_value and the rest is 0
    """
    if len(points) < 3:
        raise ValueError("At least 3 points are required for a filled curve")
    
    # Create an empty mask
    mask = np.zeros(shape, dtype=np.uint8)
    
    # Calculate center of the grid in pixel coordinates
    center_x_pixel = shape[1] // 2
    center_y_pixel = shape[0] // 2
    
    # Convert meter coordinates to pixel coordinates
    def to_pixel(point):
        # X: positive to the right from center
        # Y: positive up from center (but image coordinates go down)
        px = int(center_x_pixel + (point[0] / grid_resolution))
        py = int(center_y_pixel - (point[1] / grid_resolution))
        return (px, py) 
    
    # Find the center of the points
    center_x = sum(p[0] for p in points) / len(points)
    center_y = sum(p[1] for p in points) / len(points)
    
    # Sort points by angle around the center (ensures correct polygon ordering)
    sorted_points = sorted(points, key=lambda p: np.arctan2(p[1] - center_y, p[0] - center_x))
    
    # Convert to pixel coordinates for polygon filling
    pixel_points = [to_pixel(p) for p in sorted_points]
    
    # Simple polygon filling (always reliable)
    polygon = np.array([pixel_points], dtype=np.int32)
    simple_mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(simple_mask, polygon, fill_value)
    
    # If curves aren't enabled, return the simple polygon
    if not curves_enabled:
        return simple_mask

    # If curves are enabled, try to create a smoother shape but with safety fallbacks
    try:
        curve_mask = np.zeros(shape, dtype=np.uint8)
        
        # Draw edges between consecutive points using Bezier curves for smooth transitions
        # This creates a more controlled curve than cubic spline interpolation
        for i in range(len(sorted_points)):
            p1 = sorted_points[i]
            p2 = sorted_points[(i + 1) % len(sorted_points)]
            
            # Convert to pixel coordinates
            p1_px = to_pixel(p1)
            p2_px = to_pixel(p2)
            
            # Find the midpoint for control point calculation
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            
            # Calculate distance between points
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = np.sqrt(dx**2 + dy**2)
            
            # Calculate control points for Bezier curve (closer to endpoints for tighter curves)
            tension_factor = curve_tension * dist
            control_x = mid_x + tension_factor * (p2[1] - p1[1]) / dist
            control_y = mid_y - tension_factor * (p2[0] - p1[0]) / dist
            
            # Convert control point to pixels
            control_px = to_pixel((control_x, control_y))
            
            # Sample points along the quadratic Bezier curve
            num_samples = max(int(dist / grid_resolution * 5), 20)  # Adaptive sampling
            bezier_points = []
            
            for t in np.linspace(0, 1, num_samples):
                # Quadratic Bezier formula
                x = (1-t)**2 * p1[0] + 2*(1-t)*t * control_x + t**2 * p2[0]
                y = (1-t)**2 * p1[1] + 2*(1-t)*t * control_y + t**2 * p2[1]
                bezier_points.append(to_pixel((x, y)))
            
            # Draw the curve segment as a polyline
            if bezier_points:
                cv2.polylines(curve_mask, [np.array(bezier_points, dtype=np.int32)], 
                             False, fill_value, thickness=1)
        
        # Use morphological operations to close the shape
        kernel = np.ones((3, 3), np.uint8)
        curve_mask = cv2.morphologyEx(curve_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Use flood fill from center to fill the shape
        center_point = to_pixel((center_x, center_y))
        # Make sure center point is within bounds
        center_point = (
            min(max(center_point[0], 0), shape[1]-1),
            min(max(center_point[1], 0), shape[0]-1)
        )
        
        # Clone the mask for flood fill
        floodfill_mask = curve_mask.copy()
        h, w = floodfill_mask.shape[:2]
        flood_mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(floodfill_mask, flood_mask, center_point, fill_value)
        
        # Check if the result is reasonable
        if np.sum(floodfill_mask) > 100:  # Reasonable fill
            return floodfill_mask
        
        # Fallback 1: Try direct polygon filling with the bezier points
        bezier_polygon = np.zeros(shape, dtype=np.uint8)
        all_bezier_points = []
        
        # Sample each edge more densely
        for i in range(len(sorted_points)):
            p1 = sorted_points[i]
            p2 = sorted_points[(i + 1) % len(sorted_points)]
            
            # Calculate control points for Bezier curve
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = np.sqrt(dx**2 + dy**2)
            
            tension_factor = curve_tension * dist
            control_x = mid_x + tension_factor * (p2[1] - p1[1]) / dist
            control_y = mid_y - tension_factor * (p2[0] - p1[0]) / dist
            
            # Sample points
            num_samples = max(int(dist / grid_resolution * 2), 10)
            for t in np.linspace(0, 1, num_samples, endpoint=(i==len(sorted_points)-1)):
                x = (1-t)**2 * p1[0] + 2*(1-t)*t * control_x + t**2 * p2[0]
                y = (1-t)**2 * p1[1] + 2*(1-t)*t * control_y + t**2 * p2[1]
                all_bezier_points.append(to_pixel((x, y)))
        
        # Fill the polygon if we have enough points
        if len(all_bezier_points) > 3:
            bezier_poly = np.array([all_bezier_points], dtype=np.int32)
            cv2.fillPoly(bezier_polygon, bezier_poly, fill_value)
            
            if np.sum(bezier_polygon) > 100:  # Reasonable fill
                return bezier_polygon
    
    except Exception as e:
        # Any errors, fall back to simple polygon
        pass
        
    # Final fallback - use the reliable simple polygon filling
    return simple_mask

def line_relative_to_end_point_at_angle_dist(
    p1: tuple[float, float],      # Start point of original line (x1, y1) in meters
    p2: tuple[float, float],      # End point of original line (x2, y2) in meters
    angle: float,                 # Angle in radians relative to original line
    new_line_distance: float = None,  # Length of new line (if None, uses original length)
    distance_from_end_point: float = 0.0,  # Distance from p2 along the original line direction
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Given two points that define a line segment, create a new line segment that:
    1. Starts at a point along the original line direction from p2
    2. Has the specified length (or same as original if not specified)
    3. Is positioned at the specified angle relative to the original line
    
    Args:
        p1: Start point of original line (x1, y1) in meters
        p2: End point of original line (x2, y2) in meters
        angle: Angle in radians relative to the original line
              (0 = parallel, π/2 = perpendicular, π = opposite direction)
        new_line_distance: Length of the new line in meters (if None, uses original length)
        distance_from_end_point: Distance from p2 in the direction of the original line
                               (positive = extending beyond p2, negative = before p2)
        
    Returns:
        Two points (new_p1, new_p2) that define the new line segment
    """
    # Calculate direction vector of original line
    dir_x = p2[0] - p1[0]
    dir_y = p2[1] - p1[1]
    
    # Calculate length of original line
    length = np.sqrt(dir_x**2 + dir_y**2)
    if length < 1e-10:  # Avoid division by zero
        raise ValueError("Points are too close together")
    
    # Normalize the direction vector
    unit_dir_x = dir_x / length
    unit_dir_y = dir_y / length
    
    # Calculate perpendicular unit vector (rotate 90 degrees counterclockwise)
    unit_perp_x = -unit_dir_y
    unit_perp_y = unit_dir_x
    
    # Calculate the start point based on distance_from_end_point
    # This adjusts the starting point from p2 along the original line direction
    start_x = p2[0] + distance_from_end_point * unit_dir_x
    start_y = p2[1] + distance_from_end_point * unit_dir_y
    
    # Create the new start point
    new_p1 = (start_x, start_y)
    
    # Determine the length of the new line
    new_length = length if new_line_distance is None else new_line_distance
    
    # For exact 90-degree angles (vertical to horizontal transition)
    # This correctly handles waist points
    if abs(abs(angle) - np.pi/2) < 1e-6:
        sign = 1.0 if angle > 0 else -1.0
        new_p2 = (start_x + sign * new_length, start_y)
        return (new_p1, new_p2)
    
    # Normal case - use the standard angle-based calculation
    # Calculate the direction of the new line
    new_dir_x = unit_dir_x * np.cos(angle) + unit_perp_x * np.sin(angle)
    new_dir_y = unit_dir_y * np.cos(angle) + unit_perp_y * np.sin(angle)
    
    # Normalize the new direction vector
    new_dir_length = np.sqrt(new_dir_x**2 + new_dir_y**2)
    new_dir_x /= new_dir_length
    new_dir_y /= new_dir_length
    
    # Create the new end point
    new_p2 = (new_p1[0] + new_dir_x * new_length, new_p1[1] + new_dir_y * new_length)
    
    return (np.array(new_p1), np.array(new_p2))

def find_ratio_dist(
    p1: tuple[float, float],  # First point (x1, y1) in meters
    p2: tuple[float, float],  # Second point (x2, y2) in meters
    ratio: float,             # Ratio to multiply by the distance between points
) -> float:
    """
    Given two points that define a line segment, calculate a distance that
    has the specified ratio relative to the length of the line segment.
    
    Args:
        p1: First point (x1, y1) in meters
        p2: Second point (x2, y2) in meters
        ratio: Ratio to multiply by the distance between points
        
    Returns:
        Distance in meters that is ratio * |p2-p1|
    """
    # Calculate vector from p1 to p2
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    # Calculate length of line segment
    length = np.sqrt(dx**2 + dy**2)
    
    if length < 1e-10:  # Avoid division by zero for very close points
        raise ValueError("Points are too close together")
    
    # Calculate the distance based on the ratio
    distance = length * ratio
    
    return distance

def line_intersects_polygon(
    vertices: list[tuple[float, float]], 
    line_start: tuple[float, float], 
    line_end: tuple[float, float]
) -> bool:
    """
    Check if a line segment intersects with a polygon defined by vertices.
    A line is considered to intersect if:
    1. It crosses any edge of the polygon, OR
    2. Either endpoint of the line is inside the polygon, OR
    3. The line lies along an edge of the polygon
    
    Args:
        vertices: List of (x, y) points defining the polygon vertices in order
        line_start: (x, y) coordinates of the line start point
        line_end: (x, y) coordinates of the line end point
        
    Returns:
        True if the line segment intersects with the polygon, False otherwise
    """
    def orientation(p, q, r):
        """Calculate orientation of triplet (p, q, r)
        Returns:
           0: Collinear
           1: Clockwise
           2: Counterclockwise
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-10:
            return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or Counterclockwise
    
    def on_segment(p, q, r):
        """Check if point q lies on line segment 'pr'"""
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
    
    def do_segments_intersect(p1, q1, p2, q2):
        """Check if line segments (p1,q1) and (p2,q2) intersect"""
        # Find the four orientations needed for general and special cases
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)
        
        # General case
        if o1 != o2 and o3 != o4:
            return True
        
        # Special Cases
        # p1, q1 and p2 are collinear and p2 lies on segment p1q1
        if o1 == 0 and on_segment(p1, p2, q1):
            return True
        
        # p1, q1 and q2 are collinear and q2 lies on segment p1q1
        if o2 == 0 and on_segment(p1, q2, q1):
            return True
        
        # p2, q2 and p1 are collinear and p1 lies on segment p2q2
        if o3 == 0 and on_segment(p2, p1, q2):
            return True
        
        # p2, q2 and q1 are collinear and q1 lies on segment p2q2
        if o4 == 0 and on_segment(p2, q1, q2):
            return True
        
        return False  # No intersection
    
    # First check: Is either endpoint inside the polygon?
    if point_inside_polygon(line_start, vertices) or point_inside_polygon(line_end, vertices):
        return True
        
    # Second check: Does the line cross any edge of the polygon?
    n = len(vertices)
    for i in range(n):
        edge_start = vertices[i]
        edge_end = vertices[(i + 1) % n]
        
        if do_segments_intersect(line_start, line_end, edge_start, edge_end):
            return True
    
    return False

def point_inside_polygon(
    point: tuple[float, float], 
    vertices: list[tuple[float, float]]
) -> bool:
    """
    Check if a point is inside a polygon using the ray casting algorithm.
    
    Args:
        point: (x, y) coordinates of the point to check
        vertices: List of (x, y) points defining the polygon vertices in order
        
    Returns:
        True if the point is inside the polygon, False otherwise
    """
    x, y = point
    n = len(vertices)
    inside = False
    
    p1x, p1y = vertices[0]
    for i in range(n + 1):
        p2x, p2y = vertices[i % n]
        
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                x_intersect = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            
            if p1x == p2x or x <= x_intersect:
                inside = not inside
        
        p1x, p1y = p2x, p2y
    
    return inside

def compute_bounding_boxes(body_parts, grid, custom_labels=None):
    """
    Compute bounding boxes for body parts with improved handling of groups.
    
    Args:
        body_parts: List of body parts or groups of body parts with labels
        grid: Reference grid for sizing
        custom_labels: Optional custom labels for specific body parts
        
    Returns:
        List of bounding box dictionaries with name and coordinates
    """
    bboxes = []
    center_x_pixel = grid.shape[1] // 2
    center_y_pixel = grid.shape[0] // 2
    grid_resolution = 0.01
    
    # Process individual parts first
    individual_parts = []
    for i, part_info in enumerate(body_parts):
        if isinstance(part_info, tuple):  # Group of parts with custom label
            continue
        else:
            individual_parts.append((part_info, None))
    
    # Then process groups
    for part_info in body_parts:
        if isinstance(part_info, tuple):  # Group of parts with custom label
            parts = part_info[0]
            group_label = part_info[1]
            individual_parts.append((parts, group_label))
    
    # Now process everything
    for parts, group_label in individual_parts:
        if not isinstance(parts, list):
            parts = [parts]
            
        # Collect all vertices from all parts in the group
        all_px_values = []
        all_py_values = []
        
        for body_part in parts:
            vertices = body_part.get_vertices()
            for v in vertices:
                # Convert from meters to pixels
                px = int(center_x_pixel + (v[0] / grid_resolution))
                py = int(center_y_pixel - (v[1] / grid_resolution))
                all_px_values.append(px)
                all_py_values.append(py)
        
        # Calculate the combined bounding box
        if all_px_values:
            min_x, max_x = min(all_px_values), max(all_px_values)
            min_y, max_y = min(all_py_values), max(all_py_values)
            
            # Determine the label
            if group_label:
                name = group_label
            elif len(parts) == 1:
                name = parts[0].name
            else:
                name = "+".join([part.name for part in parts])
            
            bbox_info = {
                'name': name,
                'coords': (min_x, min_y, max_x, max_y)
            }
            bboxes.append(bbox_info)
    
    return bboxes

def process_and_filter_bboxes(all_bboxes, combined_mask, grid):
    """
    Process and filter bounding boxes with a comprehensive approach
    
    Args:
        all_bboxes: List of bounding boxes
        combined_mask: Binary mask of the body
        grid: Reference grid
        
    Returns:
        Filtered list of bounding boxes
    """
    if not all_bboxes:
        return []
        
    # First filter out sparse/noisy detections
    filtered_bboxes = filter_sparse_detections(all_bboxes, combined_mask, 
                                             min_pixel_density=0.08,  # At least 8% of box must be foreground
                                             min_pixels=30)          # Must have at least 30 foreground pixels
    
    # Filter out arms inside torsos by manually checking
    if filtered_bboxes:
        # Separate arm and torso boxes
        arm_boxes = []
        other_boxes = []
        
        for box in filtered_bboxes:
            # Check if it's an arm
            is_arm = False
            if isinstance(box, dict) and 'name' in box:
                is_arm = "arm" in box['name'].lower()
                
            if is_arm:
                arm_boxes.append(box)
            else:
                other_boxes.append(box)
        
        # Start with non-arm boxes
        result_boxes = other_boxes.copy()
        
        # Add arms only if they're not mostly inside a torso
        for arm_box in arm_boxes:
            keep_arm = True
            
            for other_box in other_boxes:
                # Check if other box is a torso
                is_torso = False
                if isinstance(other_box, dict) and 'name' in other_box:
                    is_torso = "tors" in other_box['name'].lower()
                
                if is_torso:
                    # Calculate IoU between arm and torso
                    iou = calculate_iou(arm_box, other_box)
                    if iou > 0.7:  # If arm is mostly inside torso
                        keep_arm = False
                        break
            
            if keep_arm:
                result_boxes.append(arm_box)
        
        filtered_bboxes = result_boxes
    
    # Finally use the safe filter for remaining boxes
    if filtered_bboxes:
        filtered_bboxes = safe_filter_nested_bboxes(filtered_bboxes)
    
    return filtered_bboxes

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: Bounding boxes with 'coords' key containing (x1, y1, x2, y2)
        
    Returns:
        IoU value between 0 and 1
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1['coords']
    x1_2, y1_2, x2_2, y2_2 = box2['coords']
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate areas of each box
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate IoU
    iou = intersection_area / float(box1_area)
    
    return iou

# Function to plot a body part
def plot_body_part(body_part, grid):
    # Get the vertices from the body part with their names
    vertices = []
    vertex_names = []
    for name, vertex in body_part.vertex_mapping.items():
        vertices.append(vertex)
        vertex_names.append(name)
    
    
    # Create mask using create_filled_curve_mask
    try:
        mask = create_filled_curve_mask(grid.shape, points=vertices, curves_enabled=True, curve_tension=0.1)
    except Exception as e:
        print(f"Error creating mask for {body_part.name}: {e}")
        mask = np.zeros(grid.shape, dtype=bool)
    
    # Convert body part vertices from meters to pixels
    center_x_pixel = grid.shape[1] // 2
    center_y_pixel = grid.shape[0] // 2
    grid_resolution = 0.01
    
    for i, v in enumerate(vertices):
        # Convert from meters to pixels
        px = int(center_x_pixel + (v[0] / grid_resolution))
        py = int(center_y_pixel - (v[1] / grid_resolution))
        plt.plot(px, py, 'ro')  # Red dot for each vertex
        
        # Add vertex label in red
        plt.text(px + 5, py, vertex_names[i], fontsize=8, color='red')
    
    return mask

# Function to draw bounding boxes
def draw_bounding_boxes(ax, bboxes):
    for bbox in bboxes:
        if bbox:
            min_x, min_y, max_x, max_y = bbox['coords']
            box = Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, 
                            linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(box)
            ax.text(min_x, min_y-5, f"{bbox['name']}: [{min_x},{min_y},{max_x},{max_y}]", 
                    color='blue', fontsize=8)
            
def get_class_id(name):
    """Map body part name to class ID"""
    # Handle leg parts specially
    if "leg" in name.lower() or "thigh" in name.lower():
        return YOLO_CLASS_MAPPING.index("legs")
    
    # Direct mappings
    if name == "Torso":
        return YOLO_CLASS_MAPPING.index("Torso")
    elif name == "Head":
        return YOLO_CLASS_MAPPING.index("Head")
    elif name.startswith("left_upper_arm"):
        return YOLO_CLASS_MAPPING.index("left_upper_arm")
    elif name.startswith("right_upper_arm"):
        return YOLO_CLASS_MAPPING.index("right_upper_arm")
    elif name.startswith("left_lower_arm"):
        return YOLO_CLASS_MAPPING.index("left_lower_arm")
    elif name.startswith("right_lower_arm"):
        return YOLO_CLASS_MAPPING.index("right_lower_arm")
    
    # If no match found
    raise ValueError(f"Unknown class name: {name}")         

def save_yolo_labels(bboxes, file_path, img_size):
    """
    Save bounding boxes in YOLO format
    
    Args:
        bboxes: List of bounding boxes with 'name' and 'coords' keys
        file_path: Path to save the labels
        img_size: Tuple of (width, height) for normalization
    """
    with open(file_path, 'w') as f:
        for bbox in bboxes:
            # Extract coordinates and class name
            if 'coords' in bbox and 'name' in bbox:
                x1, y1, x2, y2 = bbox['coords']
                
                # Calculate width and height (in pixels)
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                
                # Skip invalid boxes
                if width <= 0 or height <= 0:
                    continue
                
                # Calculate center point
                center_x = x1 + width / 2
                center_y = y1 + height / 2
                
                # Normalize by image dimensions (and ensure values are between 0 and 1)
                img_width, img_height = img_size[1], img_size[0]  # Reversed for width, height
                norm_center_x = max(0.001, min(0.999, center_x / img_width))
                norm_center_y = max(0.001, min(0.999, center_y / img_height))
                norm_width = max(0.001, min(0.999, width / img_width))
                norm_height = max(0.001, min(0.999, height / img_height))
                
                # Get class ID
                try:
                    class_id = get_class_id(bbox['name'])
                    # Write in YOLO format: class_id center_x center_y width height
                    f.write(f"{class_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                except ValueError as e:
                    print(f"Warning: {e} for bbox {bbox['name']}")


def safe_filter_nested_bboxes(bboxes):
    """
    Filter nested bounding boxes, ensuring we don't remove all boxes
    
    Args:
        bboxes: List of bounding boxes in the format returned by compute_bounding_boxes
        
    Returns:
        List of filtered bounding boxes
    """
    # If we have very few boxes, don't filter
    if len(bboxes) <= 2:
        return bboxes
    
    # Initialize filtered list with first box
    filtered = [bboxes[0]]
    
    # For each additional box
    for box in bboxes[1:]:
        # Extract coordinates regardless of box format 
        # (handles both class_id, coords and dictionary formats)
        if isinstance(box, tuple) and len(box) == 2:
            class_id, coords = box
        else:
            # Box is likely in another format, keep it as is
            filtered.append(box)
            continue
            
        # Check against existing filtered boxes
        should_add = True
        for existing in filtered:
            # Extract coordinates of existing box
            if isinstance(existing, tuple) and len(existing) == 2:
                existing_class, existing_coords = existing
            else:
                continue
                
            # Skip comparison if same class (we want one of each class)
            if class_id == existing_class:
                should_add = False
                break
                
            # Check for significant overlap only between different classes
            # (This is where we would implement containment logic)
                
        if should_add:
            filtered.append(box)
    
    # Ensure we keep at least one box
    if len(filtered) == 0:
        return [bboxes[0]] if bboxes else []
        
    return filtered

def filter_sparse_detections(bboxes, mask, min_pixel_density=0.1, min_pixels=20):
    """
    Filter out bounding boxes that contain mostly empty space.
    
    Args:
        bboxes: List of bounding boxes
        mask: Binary mask image (numpy array)
        min_pixel_density: Minimum ratio of foreground pixels to total box area
        min_pixels: Minimum absolute number of pixels required
        
    Returns:
        Filtered list of bounding boxes
    """
    if not bboxes:
        return []
        
    filtered_bboxes = []
    
    for box in bboxes:
        # Extract coordinates
        if isinstance(box, dict) and 'coords' in box:
            coords = box['coords']
        elif isinstance(box, tuple) and len(box) > 1 and isinstance(box[1], (list, tuple)):
            coords = box[1]
        else:
            continue
            
        # Convert to integers and ensure valid indices
        x1 = max(0, int(coords[0]))
        y1 = max(0, int(coords[1]))
        x2 = min(mask.shape[1]-1, int(coords[2]))
        y2 = min(mask.shape[0]-1, int(coords[3]))
        
        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue
            
        # Extract the region of interest from the mask
        roi = mask[y1:y2+1, x1:x2+1]
        
        # Count foreground pixels
        foreground_pixels = np.sum(roi > 0)
        total_pixels = roi.size
        
        # Calculate pixel density
        pixel_density = foreground_pixels / total_pixels
        
        # Keep the box if it meets our criteria
        if foreground_pixels >= min_pixels and pixel_density >= min_pixel_density:
            filtered_bboxes.append(box)
    
    return filtered_bboxes

def validate_body_mask(mask, min_body_pixels=1000, structure_ratio=0.7):
    """
    Validate if a body mask has enough structure to be considered valid.
    
    Args:
        mask: Binary mask of the body
        min_body_pixels: Minimum number of foreground pixels
        structure_ratio: Minimum ratio of largest component size to min_body_pixels
        
    Returns:
        Boolean indicating if the mask is valid
    """
    # Check total number of pixels
    if np.sum(mask) < min_body_pixels:
        return False
        
    # Check for connected components
    labeled_mask, num_components = ndimage.label(mask)
    component_sizes = np.bincount(labeled_mask.ravel())[1:]  # Skip background
    
    # Check if the largest component is substantial
    if len(component_sizes) == 0 or np.max(component_sizes) < min_body_pixels * structure_ratio:
        return False
        
    return True

def validate_body_parts(body_parts, grid, min_body_pixels=1000):
    """
    Create and validate mask from body parts.
    
    Args:
        body_parts: List of body parts
        grid: Reference grid for sizing
        min_body_pixels: Minimum number of foreground pixels
        
    Returns:
        Tuple of (is_valid, combined_mask, bbox_groups) or (False, None, None) if invalid
    """
    if not body_parts:
        return False, None, None
        
    # Create the combined mask
    combined_mask = np.zeros(grid.shape, dtype=bool)
    
    # Organize parts for bounding box groups
    bbox_groups = []
    leg_parts = []
    
    for part in body_parts:
        # Get mask for this part
        vertices = part.get_vertices()
        try:
            # Smooth out the head to get better detection 
            if part.name == "head":
                part_mask = create_filled_curve_mask(grid.shape, points=vertices, 
                                             curves_enabled=True, curve_tension=1.0)
            else:
                part_mask = create_filled_curve_mask(grid.shape, points=vertices, 
                                             curves_enabled=True, curve_tension=0.1)
            combined_mask = np.maximum(combined_mask, part_mask)
            
            # Organize parts for bounding box groups
            if isinstance(part, object) and hasattr(part, "name"):
                if "thigh" in part.name.lower() or "leg" in part.name.lower():
                    leg_parts.append(part)
                else:
                    bbox_groups.append(part)
                    
        except Exception as e:
            print(f"Error creating mask: {e}")
    
    # Add all leg parts as a single group if they exist
    if leg_parts:
        bbox_groups.append((leg_parts, "legs"))
    
    # Validate the mask
    if not validate_body_mask(combined_mask, min_body_pixels):
        return False, None, None
        
    return True, combined_mask, bbox_groups

def validate_processed_mask(original_mask, processed_mask, min_body_pixels=1000, retention_ratio=0.6):
    """
    Validate if a processed mask retains enough of the original structure.
    
    Args:
        original_mask: Original binary mask before processing
        processed_mask: Processed binary mask after applying noise/effects
        min_body_pixels: Minimum number of foreground pixels
        retention_ratio: Minimum ratio of pixels to retain after processing
        
    Returns:
        Boolean indicating if the processed mask is valid
    """
    # Check if we still have enough pixels after processing
    if np.sum(processed_mask) < min_body_pixels * retention_ratio:
        return False
        
    # Validate the processed mask has structure on its own
    if not validate_body_mask(processed_mask, min_body_pixels=int(min_body_pixels * retention_ratio)):
        return False
        
    return True

def calculate_box_density(box, mask):
    """
    Calculate the density of foreground pixels within a bounding box.
    
    Args:
        box: Dictionary containing 'coords' with (x1, y1, x2, y2)
        mask: Binary mask image
        
    Returns:
        Tuple of (density, foreground_pixels, total_pixels)
    """
    x1, y1, x2, y2 = box['coords']
    
    # Ensure coordinates are valid
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(mask.shape[1]-1, int(x2))
    y2 = min(mask.shape[0]-1, int(y2))
    
    # Skip invalid boxes
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0, 0
    
    # Extract region of interest
    roi = mask[y1:y2+1, x1:x2+1]
    
    # Count pixels
    foreground_pixels = np.sum(roi > 0)
    total_pixels = roi.size
    
    # Calculate density
    density = foreground_pixels / total_pixels if total_pixels > 0 else 0.0
    
    return density, foreground_pixels, total_pixels

def validate_bounding_boxes(boxes, mask, min_density=0.15, min_pixels=100, min_valid_boxes=3):
    """
    Strictly validate bounding boxes to ensure they contain actual content.
    
    Args:
        boxes: List of bounding box dictionaries
        mask: Binary mask image
        min_density: Minimum pixel density within each box
        min_pixels: Minimum absolute number of pixels in each box
        min_valid_boxes: Minimum number of valid boxes required
        
    Returns:
        Boolean indicating if boxes are valid
    """
    if not boxes or len(boxes) < min_valid_boxes:
        return False
    
    valid_boxes = 0
    total_boxes = len(boxes)
    
    # Track which parts we have valid boxes for
    valid_parts = set()
    essential_parts = {"Torso", "Head"}
    
    for box in boxes:
        name = box.get('name', '')
        density, foreground_pixels, total_pixels = calculate_box_density(box, mask)
        
        # Box must meet both density and minimum pixel criteria
        if density >= min_density and foreground_pixels >= min_pixels:
            valid_boxes += 1
            valid_parts.add(name)
    
    # Require at least one essential part
    has_essential = any(part in valid_parts for part in essential_parts)
    
    # Must have enough valid boxes and at least one essential part
    return valid_boxes >= min(min_valid_boxes, total_boxes) and has_essential

def perform_final_validation(mask, boxes, min_body_pixels=2000, min_density=0.15, min_pixels=100):
    """
    Final validation after all processing to ensure quality output.
    
    Args:
        mask: Processed binary mask
        boxes: List of bounding boxes
        min_body_pixels: Minimum total foreground pixels
        min_density: Minimum density within each box
        min_pixels: Minimum pixels within each box
        
    Returns:
        Boolean indicating if the final result is valid
    """
    # Check overall pixel count
    total_pixels = np.sum(mask)
    if total_pixels < min_body_pixels:
        return False
    
    # Check connected components 
    labeled_mask, num_components = ndimage.label(mask)
    component_sizes = np.bincount(labeled_mask.ravel())[1:]  # Skip background
    
    # Must have a dominant component
    if len(component_sizes) == 0 or np.max(component_sizes) < min_body_pixels * 0.7:
        return False
    
    # Validate all boxes have sufficient content
    return validate_bounding_boxes(boxes, mask, min_density, min_pixels)

def filter_nested_arms(all_bboxes, max_overlap_ratio=0.3):
    """
    Aggressively filter out arm boxes that overlap with the torso.
    
    Args:
        all_bboxes: List of bounding boxes
        max_overlap_ratio: Maximum allowed overlap ratio (arm area inside torso / arm area)
        
    Returns:
        Filtered list of bounding boxes with nested arms removed
    """
    if not all_bboxes:
        return []
    
    # Find torso boxes
    torso_boxes = []
    arm_boxes = []
    other_boxes = []
    
    for box in all_bboxes:
        name = box.get('name', '').lower()
        if 'torso' in name:
            torso_boxes.append(box)
        elif 'arm' in name:
            arm_boxes.append(box)
        else:
            other_boxes.append(box)
    
    # If no torso or no arms, just return the original list
    if not torso_boxes or not arm_boxes:
        return all_bboxes
    
    # Check each arm against all torsos
    kept_arms = []
    for arm_box in arm_boxes:
        keep_arm = True
        
        # Get arm box coordinates
        arm_x1, arm_y1, arm_x2, arm_y2 = arm_box['coords']
        arm_area = (arm_x2 - arm_x1) * (arm_y2 - arm_y1)
        
        for torso_box in torso_boxes:
            # Get torso box coordinates
            torso_x1, torso_y1, torso_x2, torso_y2 = torso_box['coords']
            
            # Calculate overlap area
            overlap_x1 = max(arm_x1, torso_x1)
            overlap_y1 = max(arm_y1, torso_y1)
            overlap_x2 = min(arm_x2, torso_x2)
            overlap_y2 = min(arm_y2, torso_y2)
            
            # Check if there is an overlap
            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                overlap_ratio = overlap_area / arm_area
                
                # If arm has significant overlap with torso, don't keep it
                if overlap_ratio > max_overlap_ratio:
                    keep_arm = False
                    break
        
        if keep_arm:
            kept_arms.append(arm_box)
    
    # Return filtered boxes
    return torso_boxes + kept_arms + other_boxes

if __name__ == "__main__":
    # Test the line_intersects_polygon and point_inside_polygon functions
    def test_polygon_functions():
        # Create a test polygon (irregular shape)
        polygon = [
            (1, 1),   # Bottom left
            (4, 0),   # Bottom right
            (5, 3),   # Right
            (3, 5),   # Top
            (0, 3),   # Left
        ]
        
        # Test lines
        test_lines = [
            ((0, 0), (6, 0)),     # Line below polygon (no intersection)
            ((0, 0), (2, 2)),     # Line intersecting once
            ((2, 2), (4, 4)),     # Line passing through polygon (intersects twice)
            ((0, 3), (5, 3)),     # Line along one edge
            ((2.5, 2.5), (3, 3)), # Line completely inside polygon (no intersection with edges)
            ((6, 0), (6, 6)),     # Line to the right (no intersection)
        ]
        
        # Test points
        test_points = [
            (2.5, 2.5),  # Inside
            (0, 0),      # Outside
            (1, 1),      # On vertex
            (2, 0.5),    # On edge
            (6, 3),      # Outside
        ]
        
        # Set up the plot
        plt.figure(figsize=(12, 10))
        
        # Draw the polygon
        polygon_x = [p[0] for p in polygon] + [polygon[0][0]]
        polygon_y = [p[1] for p in polygon] + [polygon[0][1]]
        plt.plot(polygon_x, polygon_y, 'b-', linewidth=2, label='Polygon')
        
        # Fill polygon with light color
        plt.fill(polygon_x, polygon_y, 'lightblue', alpha=0.3)
        
        # Test and draw each line
        for i, (start, end) in enumerate(test_lines):
            intersects = line_intersects_polygon(polygon, start, end)
            line_color = 'r' if intersects else 'g'
            line_style = '-' if intersects else '--'
            plt.plot([start[0], end[0]], [start[1], end[1]], 
                     f'{line_color}{line_style}', linewidth=2, 
                     label=f'Line {i+1}: {"Intersects" if intersects else "No intersection"}')
            
            # Add arrowhead to show direction
            plt.arrow(start[0], start[1], 
                     (end[0] - start[0]) * 0.9, (end[1] - start[1]) * 0.9,
                     head_width=0.15, head_length=0.3, fc=line_color, ec=line_color)
            
            # Add small markers for start and end points
            plt.plot(start[0], start[1], f'{line_color}o', markersize=6)
            plt.plot(end[0], end[1], f'{line_color}o', markersize=6)
        
        # Test and draw each point
        for i, point in enumerate(test_points):
            inside = point_inside_polygon(point, polygon)
            point_color = 'purple' if inside else 'orange'
            marker = '*' if inside else 'X'
            plt.plot(point[0], point[1], marker=marker, markersize=10, 
                     color=point_color, 
                     label=f'Point {i+1}: {"Inside" if inside else "Outside"}')
        
        # Add labels and legend
        plt.title('Testing line_intersects_polygon and point_inside_polygon', fontsize=14)
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right', fontsize=10)
        
        # Set equal aspect ratio and add some margin
        plt.axis('equal')
        plt.xlim(-1, 7)
        plt.ylim(-1, 7)
        
        plt.tight_layout()
        plt.savefig('polygon_intersection_test.png')
        plt.show()
        
        # Print test results
        print("\n===== TEST RESULTS =====")
        print("Line Intersection Tests:")
        for i, (start, end) in enumerate(test_lines):
            result = line_intersects_polygon(polygon, start, end)
            print(f"  Line {i+1} ({start} to {end}): {'INTERSECTS' if result else 'NO INTERSECTION'}")
        
        print("\nPoint Inside Tests:")
        for i, point in enumerate(test_points):
            result = point_inside_polygon(point, polygon)
            print(f"  Point {i+1} {point}: {'INSIDE' if result else 'OUTSIDE'}")

    # Test cases for the find_ratio_dist function
    def test_find_ratio_dist():
        test_cases = [
            ((0, 0), (3, 4), 1.0),    # Full distance (5 units)
            ((0, 0), (3, 4), 0.5),    # Half distance (2.5 units)
            ((1, 1), (5, 1), 0.75),   # 3/4 of a horizontal line (3 units)
            ((2, 3), (2, 7), 0.25),   # 1/4 of a vertical line (1 unit)
        ]
        
        print("\n===== RATIO DISTANCE TESTS =====")
        for p1, p2, ratio in test_cases:
            distance = find_ratio_dist(p1, p2, ratio)
            expected = ratio * np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            print(f"  Points {p1} to {p2}, ratio {ratio}: {distance:.2f} (expected: {expected:.2f})")
            assert abs(distance - expected) < 1e-10, "Distance calculation error!"
    
    # Test cases for line_relative_to_end_point_at_angle_dist
    def test_line_relative():
        test_cases = [
            # p1, p2, angle (degrees), new_line_distance, distance_from_end_point
            ((0, 0), (10, 0), 90, 5, 0),     # Perpendicular up from end
            ((0, 0), (0, 10), 90, 5, 0),     # Perpendicular right from end
            ((0, 0), (10, 0), 45, 7, 2),     # 45 degrees, offset 2 from end
            ((5, 5), (8, 9), -90, 4, -1),    # 90 degrees clockwise, 1 unit before end
        ]
        
        # Set up plot for line tests
        plt.figure(figsize=(10, 10))
        
        print("\n===== LINE RELATIVE TESTS =====")
        for i, (p1, p2, angle_deg, new_dist, offset) in enumerate(test_cases):
            angle_rad = math.radians(angle_deg)
            new_line = line_relative_to_end_point_at_angle_dist(
                p1, p2, angle_rad, new_dist, offset
            )
            
            # Original line (blue)
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=2)
            plt.plot(p1[0], p1[1], 'bo', markersize=6)
            plt.plot(p2[0], p2[1], 'bo', markersize=6)
            
            # New line (red)
            plt.plot([new_line[0][0], new_line[1][0]], [new_line[0][1], new_line[1][1]], 
                     'r-', linewidth=2)
            plt.plot(new_line[0][0], new_line[0][1], 'ro', markersize=6)
            plt.plot(new_line[1][0], new_line[1][1], 'ro', markersize=6)
            
            plt.annotate(f"Test {i+1}", ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2), 
                         fontsize=12, color='blue')
            plt.annotate(f"Angle: {angle_deg}°", 
                         ((new_line[0][0]+new_line[1][0])/2, (new_line[0][1]+new_line[1][1])/2), 
                         fontsize=10, color='red')
            
            print(f"  Test {i+1}: Original ({p1} to {p2}) → New line from {new_line[0]} to {new_line[1]}")
        
        plt.title('Testing line_relative_to_end_point_at_angle_dist', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('line_relative_test.png')
        plt.show()
    
    # Run all tests
    print("Running geometry function tests...")
    test_polygon_functions()
    test_find_ratio_dist()
    test_line_relative()
    print("\nAll tests completed!")
