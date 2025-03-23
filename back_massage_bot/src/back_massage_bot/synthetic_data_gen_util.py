import numpy as np
import cv2
from typing import List
from scipy.interpolate import CubicSpline

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
    
    return (new_p1, new_p2)

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