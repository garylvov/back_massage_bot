import numpy as np
import cv2
from typing import List
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import math

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