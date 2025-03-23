#!/usr/bin/env python3
# Copyright (c) 2025, Gary Lvov, Tim Bennett, Xander Ingare, Ben Yoon, Vinay Balaji
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from back_massage_bot.synthetic_data_gen_util import (create_filled_curve_mask, 
                                                        line_relative_to_end_point_at_angle_dist, 
                                                        generate_empty_grid, sample_from_params)
import math

class BodyPart:
    instances = []
    
    def __init__(self, name: str, 
                    parent = None, 
                    children = None, 
                    is_root: bool = False, 
                    vertex_mapping: dict[str, tuple[float, float]] = None, 
                    create_params: dict = None):
        self.name = name
        self.root = is_root
        self.create_params = create_params
        
        if children is None:
            self.children = []
        else:
            self.children = children
            
        if parent is not None:
            self.parent = parent
            parent.children.append(self)
        
        self.vertex_mapping = vertex_mapping if vertex_mapping is not None else {}
        if not self.vertex_mapping:
            self.populate_vertex_mapping(self.create_params)
        
        self.vertices = self.get_vertices()
    
    def get_vertices(self):
        vertices = []
        for vertex in self.vertex_mapping.values():
            vertices.append(vertex)
        return vertices
    
    def compute_semantic_mask(self):
        return {self.name: self.get_vertices()}
    
    @abstractmethod
    def populate_vertex_mapping(self, create_params: dict) -> dict[str, tuple[float, float]]:
        pass

class Torso(BodyPart):
    def __init__(self, 
                parent = None, 
                children = None,
                create_params: dict[str, float] = {"torso_length": (0.2, 0.7), 
                                                    "shoulder_from_bottom_ratio": (0.95, 1.0),
                                                    "shoulder_width_torso_length_ratio": (0.3, 0.7),
                                                    "waist_from_bottom_ratio": (0.0, .05),
                                                    "waist_width_shoulder_width_ratio": (.4, 1.0)}):
        super().__init__(name = "Torso", 
                            parent = parent, 
                            children = children, 
                            is_root = True,
                            create_params = create_params)
        
    def populate_vertex_mapping(self, create_params: dict) -> dict[str, tuple[float, float]]:
        create_params = sample_from_params(create_params)
        self.vertex_mapping = {}
        self.vertex_mapping["center_top_torso"] = np.array((0, create_params["torso_length"]))
        self.vertex_mapping["center_bottom_torso"] = np.array((0, 0))
        
        for part in ["shoulder", "waist"]:
            for idx, prefix in enumerate(["left", "right"]):
                ratio = create_params[f"{part}_from_bottom_ratio"]
                torso_length = create_params["torso_length"]
                if part == "shoulder":
                    width = create_params[f"torso_length"] * create_params[f"shoulder_width_torso_length_ratio"]
                else:
                    width = create_params[f"{part}_width_shoulder_width_ratio"] * \
                        np.linalg.norm(self.vertex_mapping["left_shoulder"] - self.vertex_mapping["right_shoulder"])
                dist = -1 * ratio * torso_length
                
                
                sign = -1 if idx == 0 else 1
                # Get both end points from the function
                _, end_point = line_relative_to_end_point_at_angle_dist(
                    p1=self.vertex_mapping["center_top_torso"], 
                    p2=self.vertex_mapping["center_bottom_torso"], 
                    angle=sign * math.pi / 2, 
                    distance_from_end_point=dist,
                    new_line_distance=width / 2
                )
                
                # Store the end point only
                self.vertex_mapping[f"{prefix}_{part}"] = np.array(end_point)
        
        return self.vertex_mapping
    
    
class Head(BodyPart):
    def __init__(self, name: str = "Head", parent = None, children = None, 
                create_params: dict[str, float] = {"head_length_to_torso_ratio": (0.28, 0.45),
                                                  "head_width_to_length_ratio": (0.7, 0.85),
                                                  "head_vertical_offset_ratio": (0.0, 0.05)}):
        super().__init__(name = name, parent = parent, children = children, is_root = False,
                        create_params = create_params)

    def populate_vertex_mapping(self, create_params: dict) -> dict[str, tuple[float, float]]:
        create_params = sample_from_params(create_params)
        parent_vertex_mapping = self.parent.vertex_mapping
        self.vertex_mapping = {}
        
        # Get parent top point (where head would normally attach)
        parent_top = parent_vertex_mapping["center_top_torso"]
        
        # Calculate torso length for reference
        torso_length = np.linalg.norm(parent_vertex_mapping["center_top_torso"] - parent_vertex_mapping["center_bottom_torso"])
        
        # Calculate vertical offset - how far down from top the head should be positioned
        vertical_offset = torso_length * create_params["head_vertical_offset_ratio"]
        
        # Calculate head dimensions
        head_length = torso_length * create_params["head_length_to_torso_ratio"]
        head_width = head_length * create_params["head_width_to_length_ratio"]
        
        # Define head bottom with potential offset from torso top
        # Move down by subtracting from y-coordinate
        head_bottom = parent_top - np.array([0, vertical_offset])
        self.vertex_mapping["center_bottom_head"] = head_bottom
        
        # Get head top (above the bottom by head_length)
        center_top_head = head_bottom + np.array([0, head_length])
        self.vertex_mapping["center_top_head"] = center_top_head
        
        # Get left and right head points at middle height
        head_middle_height = head_bottom + np.array([0, head_length/2])
        left_head = head_middle_height + np.array([-head_width/2, 0])
        right_head = head_middle_height + np.array([head_width/2, 0])
        
        self.vertex_mapping["left_head"] = left_head
        self.vertex_mapping["right_head"] = right_head
        
        return self.vertex_mapping
    
class DistalAppendage(BodyPart):
    def __init__(self, name: str, prefix: str, parent = None, children = None):
        super().__init__(name = prefix + name, parent = parent, children = children, is_root = False)

    def populate_vertex_mapping(self, create_params: dict) -> dict[str, tuple[float, float]]:
        pass

class ProximalAppendage(BodyPart):
    def __init__(self, 
                    name: str, 
                    prefix: str,
                    parent = None, children = None,
                create_params: dict = None):

        super().__init__(name = f"{prefix}_{name}", parent = parent, children = children, 
                        is_root = False, create_params = create_params)

    def populate_vertex_mapping(self, create_params: dict) -> dict[str, tuple[float, float]]:
        pass 


    

if __name__ == "__main__":
    grid = generate_empty_grid()
    
    # Create the torso
    torso = Torso()
    
    # Create the head
    head = Head(parent=torso)
    
    # Create left and right upper arms
    left_arm = ProximalAppendage(name="upper_arm", prefix="left", parent=torso)
    right_arm = ProximalAppendage(name="upper_arm", prefix="right", parent=torso)

    # # Create left and right thighs with custom leg placement
    # left_thigh = ProximalAppendage(name="leg", prefix="left", parent=torso, 
    #                               create_params={"leg_placement_ratio": (0.2, 0.25)})
    # right_thigh = ProximalAppendage(name="leg", prefix="right", parent=torso, 
    #                                create_params={"leg_placement_ratio": (0.2, 0.25)})
    
    # Display the result
    plt.figure(figsize=(10, 8))
    
    # Function to plot a body part
    def plot_body_part(body_part, grid):
        # Get the vertices from the body part with their names
        vertices = []
        vertex_names = []
        for name, vertex in body_part.vertex_mapping.items():
            vertices.append(vertex)
            vertex_names.append(name)
        
        print(f"{body_part.name} vertices with names:")
        for name, vertex in zip(vertex_names, vertices):
            print(f"  {name}: {vertex}")
        
        # Create mask using create_filled_curve_mask
        mask = create_filled_curve_mask(grid.shape, points=vertices, curves_enabled=True, curve_tension=0.1)
        
        # Convert body part vertices from meters to pixels and track min/max for bounding box
        px_values = []
        py_values = []
        center_x_pixel = grid.shape[1] // 2
        center_y_pixel = grid.shape[0] // 2
        grid_resolution = 0.01
        
        for i, v in enumerate(vertices):
            # Convert from meters to pixels
            px = int(center_x_pixel + (v[0] / grid_resolution))
            py = int(center_y_pixel - (v[1] / grid_resolution))
            px_values.append(px)
            py_values.append(py)
            plt.plot(px, py, 'ro')  # Red dot for each vertex
            
            # Add vertex label in red
            plt.text(px + 5, py, vertex_names[i], fontsize=8, color='red')
        
        # Calculate bounding box coordinates
        if px_values:  # Only create bounding box if there are vertices
            min_x, max_x = min(px_values), max(px_values)
            min_y, max_y = min(py_values), max(py_values)
            
            # Draw bounding box
            from matplotlib.patches import Rectangle
            bbox = Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, 
                            linewidth=2, edgecolor='b', facecolor='none')
            plt.gca().add_patch(bbox)
            
            # Add bounding box label
            plt.text(min_x, min_y-5, f"{body_part.name}: [{min_x},{min_y},{max_x},{max_y}]", 
                    color='blue', fontsize=10)
        
        return mask
    
    # Plot body parts
    torso_mask = plot_body_part(torso, grid)
    head_mask = plot_body_part(head, grid)
    # left_arm_mask = plot_body_part(left_arm, grid)
    # right_arm_mask = plot_body_part(right_arm, grid)
    # left_thigh_mask = plot_body_part(left_thigh, grid)
    # right_thigh_mask = plot_body_part(right_thigh, grid)
    
    # Combine masks
    combined_mask = np.maximum(torso_mask, head_mask)
    # combined_mask = np.maximum(combined_mask, left_arm_mask)
    # combined_mask = np.maximum(combined_mask, right_arm_mask)
    # combined_mask = np.maximum(combined_mask, left_thigh_mask)
    plt.imshow(combined_mask, cmap='gray')
    plt.title(f"Body Visualization")
    plt.colorbar()
    plt.show()
    