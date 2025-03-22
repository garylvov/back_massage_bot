#!/usr/bin/env python3
# Copyright (c) 2025, Gary Lvov, Tim Bennett, Xander Ingare, Ben Yoon, Vinay Balaji
# All rights reserved.
#
# SPDX-License-Identifier: MIT
import numpy as np
import matplotlib.pyplot as plt
import cv2


bbox_names = ['back', 'head', 'right_arm', 'left_arm', 'right_leg', 'left_leg', 'glutes']

def draw_limb(image, joint1_x, joint1_y, angle1_rad, angle2_rad, 
              upper_length, lower_length, width, temp_scale, is_leg=False):
    """Draw a limb (arm or leg) with upper and lower sections."""
    # Calculate middle joint position
    joint2_x = int(joint1_x + upper_length * temp_scale * 
                  (np.sin(angle1_rad) if is_leg else np.cos(angle1_rad)))
    joint2_y = int(joint1_y + upper_length * temp_scale * 
                  (np.cos(angle1_rad) if is_leg else np.sin(angle1_rad)))
    
    # Calculate end joint position
    end_angle = angle1_rad - np.pi + angle2_rad if is_leg else angle1_rad + np.pi - angle2_rad
    joint3_x = int(joint2_x + lower_length * temp_scale * 
                  (np.sin(end_angle) if is_leg else np.cos(end_angle)))
    joint3_y = int(joint2_y + lower_length * temp_scale * 
                  (np.cos(end_angle) if is_leg else np.sin(end_angle)))
    
    # Draw upper section
    cv2.line(image, (joint1_x, joint1_y), (joint2_x, joint2_y), 255, width * temp_scale)
    
    # Draw lower section
    cv2.line(image, (joint2_x, joint2_y), (joint3_x, joint3_y), 255, width * temp_scale)
    
    # Return joint positions for bounding box calculation
    return np.array([[joint1_x, joint1_y], [joint2_x, joint2_y], [joint3_x, joint3_y]])

def calculate_bbox(points, temp_scale, image_size, class_id):
    """Calculate bounding box from a set of points."""
    # Scale points back to original resolution
    points_scaled = points / temp_scale
    
    # Find min/max coordinates
    x_min = np.min(points_scaled[:, 0])
    y_min = np.min(points_scaled[:, 1])
    x_max = np.max(points_scaled[:, 0])
    y_max = np.max(points_scaled[:, 1])
    
    # Calculate dimensions and center
    width = x_max - x_min
    height = y_max - y_min
    center_x = x_min + width/2
    center_y = y_min + height/2
    
    # Create bbox dictionary
    bbox = {
        'class': class_id,
        'x_center': center_x / image_size[1],
        'y_center': center_y / image_size[0],
        'width': width / image_size[1],
        'height': height / image_size[0]
    }
    
    return bbox

def generate_synthetic_top_down_massage_occupancy_grid(image_size: tuple[int, int] = (800, 800), 
                    grid_resolution: float = 0.01,  # meters per pixel
                    torso_center_bounds: tuple[float, float] = (0.0, 0.1),  # meters from center
                    head_width_bounds: tuple[float, float] = (0.12, 0.18),  # meters
                    head_length_bounds: tuple[float, float] = (0.18, 0.25),  # meters
                    neck_width_bounds: tuple[float, float] = (0.05, 0.08),  # meters
                    neck_length_bounds: tuple[float, float] = (0.05, 0.1),  # meters
                    neck_angle_bounds: tuple[float, float] = (-10, 10),  # degrees
                    shoulder_width_bounds: tuple[float, float] = (0.38, 0.45),  # meters
                    waist_width_bounds: tuple[float, float] = (0.28, 0.35),  # meters
                    torso_length_bounds: tuple[float, float] = (0.45, 0.55),  # meters
                    glutes_width_bounds: tuple[float, float] = (0.28, 0.35),  # meters
                    glutes_length_bounds: tuple[float, float] = (0.15, 0.22),  # meters
                    arm_width_bounds: tuple[float, float] = (0.06, 0.09),  # meters
                    arm_upper_length_bounds: tuple[float, float] = (0.25, 0.35),  # meters
                    arm_lower_length_bounds: tuple[float, float] = (0.25, 0.35),  # meters
                    arm_shoulder_angle_bounds: tuple[float, float] = (10, 45),  # degrees
                    arm_elbow_angle_bounds: tuple[float, float] = (90, 170),  # degrees
                    leg_width_bounds: tuple[float, float] = (0.07, 0.1),  # meters
                    leg_upper_length_bounds: tuple[float, float] = (0.35, 0.45),  # meters
                    leg_lower_length_bounds: tuple[float, float] = (0.35, 0.45),  # meters
                    leg_hip_angle_bounds: tuple[float, float] = (-10, 30),  # degrees
                    leg_knee_angle_bounds: tuple[float, float] = (90, 170),  # degrees
                    populated_cell_single_decimation_prob: float = 0.05,
                    empty_cell_single_addition_prob: float = 0.005,
                    empty_cell_cluster_addition_prob: float = 0.2,
                    empty_cell_cluster_addition_num_pixel_bounds: tuple[int, int] = (5, 20),
                    border_decimation_width: float = 0.02,  # meters
                    border_decimation_prob: float = 0.7,
                    debug_viz = False) -> tuple[np.ndarray, list[dict]]:
    """Generate top down synthetic occupancy grid of the human body for a massage bot. Bake in some noise from the jump.
    This could be done as part of an augmentation step in the training pipeline, but for the sake of convenience, we are doing it here.
    
    All dimension bounds are in meters and converted to pixels based on grid_resolution.
    """
    # Create empty image
    image = np.zeros(image_size, dtype=np.uint8)
    
    # Calculate center of image in pixels
    center_x = image_size[1] // 2
    center_y = image_size[0] // 2
    
    # Function to convert meters to pixels
    def m_to_px(meters):
        return int(meters / grid_resolution)
    
    # Random torso center offset from center (in meters, then converted to pixels)
    torso_center_offset_x = np.random.uniform(torso_center_bounds[0], torso_center_bounds[1])
    torso_center_x = center_x + m_to_px(torso_center_offset_x)
    torso_center_y = center_y  # Keep y centered
    
    # Generate body part dimensions (convert from meters to pixels)
    head_width = m_to_px(np.random.uniform(head_width_bounds[0], head_width_bounds[1]))
    head_length = m_to_px(np.random.uniform(head_length_bounds[0], head_length_bounds[1]))
    
    neck_width = m_to_px(np.random.uniform(neck_width_bounds[0], neck_width_bounds[1]))
    neck_length = m_to_px(np.random.uniform(neck_length_bounds[0], neck_length_bounds[1]))
    neck_angle = np.random.uniform(neck_angle_bounds[0], neck_angle_bounds[1])
    
    shoulder_width = m_to_px(np.random.uniform(shoulder_width_bounds[0], shoulder_width_bounds[1]))
    waist_width = m_to_px(np.random.uniform(waist_width_bounds[0], waist_width_bounds[1]))
    torso_length = m_to_px(np.random.uniform(torso_length_bounds[0], torso_length_bounds[1]))
    
    glutes_width = m_to_px(np.random.uniform(glutes_width_bounds[0], glutes_width_bounds[1]))
    glutes_length = m_to_px(np.random.uniform(glutes_length_bounds[0], glutes_length_bounds[1]))
    
    arm_width = m_to_px(np.random.uniform(arm_width_bounds[0], arm_width_bounds[1]))
    arm_upper_length = m_to_px(np.random.uniform(arm_upper_length_bounds[0], arm_upper_length_bounds[1]))
    arm_lower_length = m_to_px(np.random.uniform(arm_lower_length_bounds[0], arm_lower_length_bounds[1]))
    
    # Arm angles (in degrees)
    right_arm_shoulder_angle = np.random.uniform(arm_shoulder_angle_bounds[0], arm_shoulder_angle_bounds[1])
    left_arm_shoulder_angle = np.random.uniform(arm_shoulder_angle_bounds[0], arm_shoulder_angle_bounds[1])
    right_arm_elbow_angle = np.random.uniform(arm_elbow_angle_bounds[0], arm_elbow_angle_bounds[1])
    left_arm_elbow_angle = np.random.uniform(arm_elbow_angle_bounds[0], arm_elbow_angle_bounds[1])
    
    leg_width = m_to_px(np.random.uniform(leg_width_bounds[0], leg_width_bounds[1]))
    leg_upper_length = m_to_px(np.random.uniform(leg_upper_length_bounds[0], leg_upper_length_bounds[1]))
    leg_lower_length = m_to_px(np.random.uniform(leg_lower_length_bounds[0], leg_lower_length_bounds[1]))
    
    # Leg angles (in degrees)
    right_leg_hip_angle = np.random.uniform(leg_hip_angle_bounds[0], leg_hip_angle_bounds[1])
    left_leg_hip_angle = np.random.uniform(leg_hip_angle_bounds[0], leg_hip_angle_bounds[1])
    right_leg_knee_angle = np.random.uniform(leg_knee_angle_bounds[0], leg_knee_angle_bounds[1])
    left_leg_knee_angle = np.random.uniform(leg_knee_angle_bounds[0], leg_knee_angle_bounds[1])
    
    # Initialize bounding boxes list
    bboxes = []
    
    # Create a temporary drawing image with higher resolution for anti-aliasing
    temp_scale = 4  # Scale factor for temporary image
    temp_image = np.zeros((image_size[0] * temp_scale, image_size[1] * temp_scale), dtype=np.uint8)
    
    # Draw torso (back)
    torso_top = int(torso_center_y - torso_length/2) * temp_scale
    torso_bottom = int(torso_center_y + torso_length/2) * temp_scale
    torso_center_x_temp = torso_center_x * temp_scale
    torso_center_y_temp = torso_center_y * temp_scale
    
    # Trapezoid shape for torso
    torso_points = np.array([
        [int(torso_center_x_temp - shoulder_width/2 * temp_scale), torso_top],
        [int(torso_center_x_temp + shoulder_width/2 * temp_scale), torso_top],
        [int(torso_center_x_temp + waist_width/2 * temp_scale), torso_bottom],
        [int(torso_center_x_temp - waist_width/2 * temp_scale), torso_bottom]
    ])
    
    # Draw torso
    cv2.fillPoly(temp_image, [torso_points], 255)
    
    # Add torso bounding box
    torso_bbox = {
        'class': 0,  # 'back'
        'x_center': torso_center_x / image_size[1],
        'y_center': torso_center_y / image_size[0],
        'width': shoulder_width / image_size[1],
        'height': torso_length / image_size[0]
    }
    bboxes.append(torso_bbox)
    
    # Draw neck
    neck_bottom_center_x = torso_center_x * temp_scale
    neck_bottom_center_y = torso_top
    neck_angle_rad = np.radians(neck_angle)
    neck_top_center_x = int(neck_bottom_center_x + neck_length * temp_scale * np.sin(neck_angle_rad))
    neck_top_center_y = int(neck_bottom_center_y - neck_length * temp_scale * np.cos(neck_angle_rad))
    
    # Neck points
    neck_points = np.array([
        [int(neck_bottom_center_x - neck_width/2 * temp_scale), neck_bottom_center_y],
        [int(neck_bottom_center_x + neck_width/2 * temp_scale), neck_bottom_center_y],
        [int(neck_top_center_x + neck_width/2 * temp_scale), neck_top_center_y],
        [int(neck_top_center_x - neck_width/2 * temp_scale), neck_top_center_y]
    ])
    
    # Draw neck
    cv2.fillPoly(temp_image, [neck_points], 255)
    
    # Draw head
    head_center_x = neck_top_center_x
    head_center_y = neck_top_center_y - head_length/2 * temp_scale
    
    # Draw head as ellipse
    cv2.ellipse(temp_image, 
                (int(head_center_x), int(head_center_y)), 
                (int(head_width/2 * temp_scale), int(head_length/2 * temp_scale)), 
                0, 0, 360, 255, -1)
    
    # Add head bounding box
    head_bbox = {
        'class': 1,  # 'head'
        'x_center': (head_center_x / temp_scale) / image_size[1],
        'y_center': (head_center_y / temp_scale) / image_size[0],
        'width': head_width / image_size[1],
        'height': head_length / image_size[0]
    }
    bboxes.append(head_bbox)
    
    # Draw glutes
    glutes_center_x = torso_center_x * temp_scale
    glutes_center_y = torso_bottom + glutes_length/2 * temp_scale
    
    # Draw glutes as ellipse
    cv2.ellipse(temp_image, 
                (int(glutes_center_x), int(glutes_center_y)), 
                (int(glutes_width/2 * temp_scale), int(glutes_length/2 * temp_scale)), 
                0, 0, 360, 255, -1)
    
    # Add glutes bounding box
    glutes_bbox = {
        'class': 6,  # 'glutes'
        'x_center': (glutes_center_x / temp_scale) / image_size[1],
        'y_center': (glutes_center_y / temp_scale) / image_size[0],
        'width': glutes_width / image_size[1],
        'height': glutes_length / image_size[0]
    }
    bboxes.append(glutes_bbox)
    
    # Draw right arm
    right_shoulder_x = int(torso_center_x_temp + shoulder_width/2 * temp_scale)
    right_shoulder_y = torso_top
    
    # Convert angles to radians
    right_arm_shoulder_angle_rad = np.radians(right_arm_shoulder_angle)
    right_arm_elbow_angle_rad = np.radians(right_arm_elbow_angle)
    
    # Draw the right arm using the helper function
    right_arm_points = draw_limb(
        temp_image, 
        right_shoulder_x, right_shoulder_y, 
        right_arm_shoulder_angle_rad, right_arm_elbow_angle_rad,
        arm_upper_length, arm_lower_length, 
        arm_width, temp_scale, 
        is_leg=False
    )
    
    # Calculate right arm bounding box using the helper function
    right_arm_bbox = calculate_bbox(right_arm_points, temp_scale, image_size, 2)  # 'right_arm'
    bboxes.append(right_arm_bbox)
    
    # Draw left arm
    left_shoulder_x = int(torso_center_x_temp - shoulder_width/2 * temp_scale)
    left_shoulder_y = torso_top
    
    # Convert angles to radians (mirror for left side)
    left_arm_shoulder_angle_rad = np.radians(180 - left_arm_shoulder_angle)
    left_arm_elbow_angle_rad = np.radians(left_arm_elbow_angle)
    
    # Draw the left arm using the helper function
    left_arm_points = draw_limb(
        temp_image, 
        left_shoulder_x, left_shoulder_y, 
        left_arm_shoulder_angle_rad, left_arm_elbow_angle_rad,
        arm_upper_length, arm_lower_length, 
        arm_width, temp_scale, 
        is_leg=False
    )
    
    # Calculate left arm bounding box using the helper function
    left_arm_bbox = calculate_bbox(left_arm_points, temp_scale, image_size, 3)  # 'left_arm'
    bboxes.append(left_arm_bbox)
    
    # Draw right leg
    right_hip_x = int(torso_center_x_temp + waist_width/4 * temp_scale)
    right_hip_y = torso_bottom
    
    # Convert angles to radians
    right_leg_hip_angle_rad = np.radians(right_leg_hip_angle)
    right_leg_knee_angle_rad = np.radians(right_leg_knee_angle)
    
    # Draw the right leg using the helper function
    right_leg_points = draw_limb(
        temp_image, 
        right_hip_x, right_hip_y, 
        right_leg_hip_angle_rad, right_leg_knee_angle_rad,
        leg_upper_length, leg_lower_length, 
        leg_width, temp_scale, 
        is_leg=True
    )
    
    # Calculate right leg bounding box using the helper function
    right_leg_bbox = calculate_bbox(right_leg_points, temp_scale, image_size, 4)  # 'right_leg'
    bboxes.append(right_leg_bbox)
    
    # Draw left leg
    left_hip_x = int(torso_center_x_temp - waist_width/4 * temp_scale)
    left_hip_y = torso_bottom
    
    # Convert angles to radians (mirror for left side)
    left_leg_hip_angle_rad = np.radians(-left_leg_hip_angle)
    left_leg_knee_angle_rad = np.radians(left_leg_knee_angle)
    
    # Draw the left leg using the helper function
    left_leg_points = draw_limb(
        temp_image, 
        left_hip_x, left_hip_y, 
        left_leg_hip_angle_rad, left_leg_knee_angle_rad,
        leg_upper_length, leg_lower_length, 
        leg_width, temp_scale, 
        is_leg=True
    )
    
    # Calculate left leg bounding box using the helper function
    left_leg_bbox = calculate_bbox(left_leg_points, temp_scale, image_size, 5)  # 'left_leg'
    bboxes.append(left_leg_bbox)
    
    # Resize the temporary image back to original size with anti-aliasing
    image = cv2.resize(temp_image, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
    
    # Threshold to create binary image
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Apply border decimation
    # First, find the edges of the body
    edges = cv2.Canny(image, 50, 150)
    
    # Dilate the edges to create a border region
    border_width_px = m_to_px(border_decimation_width)
    kernel = np.ones((border_width_px, border_width_px), np.uint8)
    border_region = cv2.dilate(edges, kernel, iterations=1) & image
    
    # Randomly decimate pixels in the border region
    border_indices = np.where(border_region > 0)
    for i in range(len(border_indices[0])):
        if np.random.random() < border_decimation_prob:
            y, x = border_indices[0][i], border_indices[1][i]
            image[y, x] = 0
    
    # Apply noise to the image
    # 1. Remove some populated cells
    populated_indices = np.where(image > 0)
    num_populated = len(populated_indices[0])
    
    for i in range(num_populated):
        if np.random.random() < populated_cell_single_decimation_prob:
            y, x = populated_indices[0][i], populated_indices[1][i]
            image[y, x] = 0
    
    # 2. Add some random single pixels
    for _ in range(int(image_size[0] * image_size[1] * empty_cell_single_addition_prob)):
        x = np.random.randint(0, image_size[1])
        y = np.random.randint(0, image_size[0])
        image[y, x] = 255
    
    # 3. Add some random clusters of pixels
    if np.random.random() < empty_cell_cluster_addition_prob:
        num_clusters = np.random.randint(1, 5)
        for _ in range(num_clusters):
            cluster_center_x = np.random.randint(0, image_size[1])
            cluster_center_y = np.random.randint(0, image_size[0])
            num_pixels = np.random.randint(empty_cell_cluster_addition_num_pixel_bounds[0], 
                                          empty_cell_cluster_addition_num_pixel_bounds[1])
            
            for _ in range(num_pixels):
                offset_x = np.random.randint(-10, 11)
                offset_y = np.random.randint(-10, 11)
                x = min(max(0, cluster_center_x + offset_x), image_size[1] - 1)
                y = min(max(0, cluster_center_y + offset_y), image_size[0] - 1)
                image[y, x] = 255
    
    # Visualize if debug mode is on
    if debug_viz:
        visualize_occupancy_grid(image, bboxes, image_size, bbox_names)
    
    return image, bboxes

def visualize_occupancy_grid(image, bboxes, image_size, bbox_names):
    """Visualize occupancy grid with bounding boxes."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    
    # Draw bounding boxes
    for bbox in bboxes:
        class_id = bbox['class']
        x_center = bbox['x_center'] * image_size[1]
        y_center = bbox['y_center'] * image_size[0]
        width = bbox['width'] * image_size[1]
        height = bbox['height'] * image_size[0]
        
        # Calculate corners
        x_min = int(x_center - width/2)
        y_min = int(y_center - height/2)
        
        # Draw rectangle
        plt.gca().add_patch(plt.Rectangle((x_min, y_min), width, height, 
                                         fill=False, edgecolor='red', linewidth=2))
        
        # Add label with semi-transparent background (rgba with alpha=0.5)
        plt.text(x_min, y_min - 5, bbox_names[class_id], 
                 color='white', fontsize=10, backgroundcolor=(1, 0, 0, 0.5))
    
    plt.title('Synthetic Body Occupancy Grid with Bounding Boxes')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_synthetic_top_down_massage_occupancy_grid(debug_viz=True)