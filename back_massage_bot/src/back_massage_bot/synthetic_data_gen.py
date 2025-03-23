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
from scipy import ndimage

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
        
        # Add this instance to the global list
        BodyPart.instances.append(self)
        
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

    @staticmethod
    def find_paired_part(name, prefix):
        """Find a paired body part (e.g., left_leg if right_leg is provided)"""
        other_prefix = "left" if prefix == "right" else "right"
        paired_name = name.replace(f"{prefix}_", f"{other_prefix}_")
        
        for part in BodyPart.instances:
            if part.name == paired_name:
                return part
        
        return None

class ProximalAppendage(BodyPart):
    def __init__(self, 
                    name: str, 
                    prefix: str,
                    avoid_collisions_with: list[BodyPart] | None = None,
                    parent = None, children = None,
                    create_params: dict = None,
                    variation_factor: float = 0.05):
        self.prefix = prefix
        self.variation_factor = variation_factor
        if avoid_collisions_with is not None:
            self.avoid_collisions_with = avoid_collisions_with
        else:
            self.avoid_collisions_with = []
            
        super().__init__(name = f"{prefix}_{name}", parent = parent, children = children, 
                        is_root = False, create_params = create_params)

    def get_sampled_params(self, create_params):
        """Get parameters with possible variations from a paired appendage"""
        # Check if there's already a paired appendage with parameters
        paired_appendage = BodyPart.find_paired_part(self.name, self.prefix)
        
        if paired_appendage is not None and hasattr(paired_appendage, 'sampled_params'):
            # Use parameters from paired appendage with slight variations
            base_params = paired_appendage.sampled_params
            sampled_params = {}
            
            # Apply slight variations to each parameter
            for key, value in base_params.items():
                if "angle" in key:
                    # For angles, we need to be careful with the sign
                    # Keep the sign but vary the magnitude slightly
                    sign = 1 if value > 0 else -1
                    magnitude = abs(value)
                    # Add variation within +/- variation_factor percentage
                    variation = 1.0 + (np.random.random() * 2 - 1) * self.variation_factor
                    sampled_params[key] = sign * magnitude * variation
                else:
                    # For regular parameters, add a small variation
                    variation = 1.0 + (np.random.random() * 2 - 1) * self.variation_factor
                    sampled_params[key] = value * variation
        else:
            # Sample new parameters
            sampled_params = sample_from_params(create_params)
        
        # Store the parameters for possible use by future paired appendages
        self.sampled_params = sampled_params
        return sampled_params

    def populate_vertex_mapping(self, create_params: dict) -> dict[str, tuple[float, float]]:
        """Base implementation for proximal appendages"""
        # Get sampled parameters with possible variations from paired appendage
        sampled_params = self.get_sampled_params(create_params)
        
        # Derived classes should implement their specific vertex mapping logic
        return self.calculate_vertices(sampled_params)

    def calculate_vertices(self, sampled_params: dict) -> dict[str, tuple[float, float]]:
        """To be implemented by derived classes to calculate specific vertices"""
        pass

class DistalAppendage(ProximalAppendage):
    """Base class for appendages that connect to other appendages (like lower arms and lower legs)"""
    
    def __init__(self,
                name: str,
                prefix: str,
                parent = None, 
                children = None,
                joint_name: str = None,             # e.g., "elbow" or "knee"
                endpoint_name: str = None,          # e.g., "wrist" or "ankle"
                parent_joint_name: str = None,      # e.g., "shoulder" or "hip"
                inner_point_name: str = None,       # e.g., "inner_lower_arm" or "inner_lower_thigh"
                outer_point_name: str = None,       # e.g., "outer_lower_arm" or "outer_lower_thigh"
                proximal_name: str = None,          # e.g., "proximal_forearm" or "proximal_lower_leg"
                distal_name: str = None,            # e.g., "distal_forearm" or "distal_lower_leg" 
                bulge_name: str = None,             # e.g., "forearm_bulge" or "calf"
                create_params: dict = None,
                variation_factor: float = 0.05):
        
        self.joint_name = joint_name
        self.endpoint_name = endpoint_name
        self.parent_joint_name = parent_joint_name
        self.inner_point_name = inner_point_name
        self.outer_point_name = outer_point_name
        self.proximal_name = proximal_name
        self.distal_name = distal_name
        self.bulge_name = bulge_name
        
        super().__init__(name = name, prefix = prefix, parent = parent, children = children, 
                        create_params = create_params, variation_factor = variation_factor)
    
    def calculate_vertices(self, sampled_params: dict) -> dict[str, tuple[float, float]]:
        # Get the joint point from the parent (e.g., elbow or knee)
        joint_point = self.parent.vertex_mapping[f"{self.prefix}_{self.joint_name}"]
        
        # Get the parent joint point for reference (e.g., shoulder or hip)
        parent_joint_point = self.parent.vertex_mapping[f"{self.prefix}_{self.parent_joint_name}"]
        
        # Calculate parent length for reference
        parent_length = np.linalg.norm(parent_joint_point - joint_point)
        
        # Calculate joint width (distance between inner and outer points)
        inner_point = self.parent.vertex_mapping[f"{self.prefix}_{self.inner_point_name}"]
        outer_point = self.parent.vertex_mapping[f"{self.prefix}_{self.outer_point_name}"]
        joint_width = np.linalg.norm(inner_point - outer_point)
        
        self.vertex_mapping = {}
        # Store the joint point
        self.vertex_mapping[f"{self.prefix}_{self.joint_name}"] = joint_point
        
        # Determine the sign for the angle based on which side we're on
        sign = 1 if self.prefix == "right" else -1
        
        # Get the angle parameter - these will differ by class (forearm vs lower leg)
        angle_param = [k for k in sampled_params.keys() if "angle" in k][0]
        
        # Calculate the endpoint position (wrist/ankle)
        _, endpoint = line_relative_to_end_point_at_angle_dist(
            p1=parent_joint_point,
            p2=joint_point,
            angle=sign * sampled_params[angle_param],
            distance_from_end_point=0,  # Start from joint
            new_line_distance=sampled_params[[k for k in sampled_params.keys() if "length" in k][0]] * parent_length
        )
        
        self.vertex_mapping[f"{self.prefix}_{self.endpoint_name}"] = endpoint
        
        # Add outer and inner points
        for loc in ["outer", "inner"]:
            # Need to adjust angle based on which side and which location
            if self.name == "lower_arm":
                # Special case for forearm
                if (self.prefix == "left" and loc == "outer") or (self.prefix == "right" and loc == "inner"):
                    angle = 90
                else:
                    angle = -90
            else:
                # For legs and other appendages
                angle = -90 if loc == "outer" else 90
            
            # Calculate the proximal (near joint) width
            central_proximal, outer_proximal = line_relative_to_end_point_at_angle_dist(
                p1=joint_point,
                p2=endpoint,
                angle=angle,
                distance_from_end_point=-1 * sampled_params[[k for k in sampled_params.keys() if "proximal" in k and "placement" in k][0]] * np.linalg.norm(joint_point - endpoint),
                new_line_distance=sampled_params[[k for k in sampled_params.keys() if "proximal" in k and "width" in k][0]] * joint_width / 2
            )
            self.vertex_mapping[f"{self.prefix}_{loc}_{self.proximal_name}"] = outer_proximal
            
            # Calculate the distal (near endpoint) width
            _, outer_distal = line_relative_to_end_point_at_angle_dist(
                p1=joint_point,
                p2=endpoint,
                angle=angle,
                distance_from_end_point=-1 * sampled_params[[k for k in sampled_params.keys() if "distal" in k and "placement" in k][0]] * np.linalg.norm(joint_point - endpoint),
                new_line_distance=sampled_params[[k for k in sampled_params.keys() if "distal" in k and "width" in k][0]] * np.linalg.norm(outer_proximal - central_proximal)
            )
            self.vertex_mapping[f"{self.prefix}_{loc}_{self.distal_name}"] = outer_distal
            
            # Add the bulge point (a point that's wider than both the joint and endpoint)
            bulge_param = [k for k in sampled_params.keys() if "bulge" in k or "calf" in k][0]
            bulge_factor = 1.05 if self.name == "lower_arm" else 1.1  # Slightly larger bulge for calf
            
            _, bulge_point = line_relative_to_end_point_at_angle_dist(
                p1=joint_point,
                p2=endpoint,
                angle=angle,
                distance_from_end_point=-1 * sampled_params[bulge_param] * np.linalg.norm(joint_point - endpoint),
                new_line_distance=bulge_factor * np.linalg.norm(outer_proximal - central_proximal)
            )
            self.vertex_mapping[f"{self.prefix}_{loc}_{self.bulge_name}"] = bulge_point
        
        return self.vertex_mapping

class Torso(BodyPart):
    def __init__(self, 
                parent = None, 
                children = None,
                create_params: dict[str, float] = {"torso_length": (0.2, 0.7), 
                                                    "shoulder_from_bottom_ratio": (0.95, 1.0),
                                                    "shoulder_width_torso_length_ratio": (0.3, 0.9),
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
    
class Thigh(ProximalAppendage):
    def __init__(self, name: str = "leg", prefix: str = None, parent = None, children = None, 
                create_params: dict[str, float] = {"hip_placement_ratio_between_waist_center": (0.0, .3),
                                                    "thigh_length_to_torso_length_ratio": (0.3, .7),
                                                    "proximal_thigh_width_to_torso_width_ratio": (0.2, 0.4),
                                                    "distal_thigh_width_to_proximal_thigh_width_ratio": (0.7, 1.0),
                                                    "proximal_thigh_placement_ratio_above_knee": (0.9, 1.0),
                                                    "distal_thigh_placement_ratio_above_knee": (0.0, 0.1),
                                                    "angle_between_thigh_and_waist_line": (math.radians(-70), math.radians(-110))},
                variation_factor: float = 0.05):
        super().__init__(name = name, prefix = prefix, parent = parent, children = children, 
                        create_params = create_params, variation_factor = variation_factor)
    
    def calculate_vertices(self, sampled_params: dict) -> dict[str, tuple[float, float]]:
        waist_point = self.parent.vertex_mapping[f"{self.prefix}_waist"]
        other_prefix = "left" if self.prefix == "right" else "right"
        other_waist_point = self.parent.vertex_mapping[f"{other_prefix}_waist"]
        
        # Calculate the midpoint between waist points
        midpoint = (waist_point + other_waist_point) / 2

        self.vertex_mapping = {}
        sign = 1 if self.prefix == "right" else -1
        hip, knee = line_relative_to_end_point_at_angle_dist(
            p1=midpoint,
            p2=waist_point,
            angle=sign * sampled_params["angle_between_thigh_and_waist_line"],
            distance_from_end_point=-1 * sampled_params["hip_placement_ratio_between_waist_center"] * np.linalg.norm(midpoint - waist_point),
            new_line_distance=sampled_params["thigh_length_to_torso_length_ratio"] * np.linalg.norm(self.parent.vertex_mapping["center_top_torso"] - midpoint)
        )
        
        self.vertex_mapping[f"{self.prefix}_hip"] = hip
        self.vertex_mapping[f"{self.prefix}_knee"] = knee
        
        for loc in ["outer", "inner"]:
            angle = -90 if loc == "outer" else 90
            # Proximal width
            central_upper_thigh, outer_upper_thigh = line_relative_to_end_point_at_angle_dist(
            p1=hip,
            p2=knee,
            angle=angle,
            distance_from_end_point=-1 * sampled_params["proximal_thigh_placement_ratio_above_knee"] * np.linalg.norm(hip - knee) ,
            new_line_distance=sampled_params["proximal_thigh_width_to_torso_width_ratio"] * np.linalg.norm(midpoint - waist_point)
            )
            self.vertex_mapping[f"{self.prefix}_{loc}_upper_thigh"] = outer_upper_thigh
            
            # Distal width
            _, outer_upper_thigh = line_relative_to_end_point_at_angle_dist(
            p1=hip,
            p2=knee,
            angle=angle,
            distance_from_end_point=-1 * sampled_params["distal_thigh_placement_ratio_above_knee"] * np.linalg.norm(hip - knee) ,
            new_line_distance=sampled_params["distal_thigh_width_to_proximal_thigh_width_ratio"] * np.linalg.norm(outer_upper_thigh - central_upper_thigh)
            )
            self.vertex_mapping[f"{self.prefix}_{loc}_lower_thigh"] = outer_upper_thigh
        
        return self.vertex_mapping

class UpperArm(ProximalAppendage):
    def __init__(self, name: str = "upper_arm", prefix: str = None, parent = None, children = None, 
                create_params: dict[str, float] = {"shoulder_placement_ratio": (0.8, 1.0),
                                                   "arm_length_to_torso_length_ratio": (0.35, 0.55),
                                                   "proximal_arm_width_to_torso_width_ratio": (0.15, 0.25),
                                                   "distal_arm_width_to_proximal_arm_width_ratio": (0.75, 0.95),
                                                   "proximal_arm_placement_ratio_above_elbow": (0.9, 1.0),
                                                   "distal_arm_placement_ratio_above_elbow": (0.0, 0.1),
                                                   "angle_between_arm_and_torso": (math.radians(10), math.radians(50))},
                variation_factor: float = 0.05):
        super().__init__(name = name, prefix = prefix, parent = parent, children = children, 
                        create_params = create_params, variation_factor = variation_factor)
    
    def calculate_vertices(self, sampled_params: dict) -> dict[str, tuple[float, float]]:
        # Get the shoulder point from the parent (torso)
        shoulder_point = self.parent.vertex_mapping[f"{self.prefix}_shoulder"]
        
        # Get the center top torso point for reference
        center_top = self.parent.vertex_mapping["center_top_torso"]
        
        # Calculate torso length and width for reference
        torso_length = np.linalg.norm(self.parent.vertex_mapping["center_top_torso"] - 
                                     self.parent.vertex_mapping["center_bottom_torso"])
        torso_width = np.linalg.norm(self.parent.vertex_mapping["left_shoulder"] - 
                                    self.parent.vertex_mapping["right_shoulder"])
        
        self.vertex_mapping = {}
        # Store the shoulder point
        self.vertex_mapping[f"{self.prefix}_shoulder"] = shoulder_point
        
        # Determine the sign for the angle based on which side we're on
        sign = -1 if self.prefix == "right" else 1
        
        # Calculate the elbow position
        _, elbow = line_relative_to_end_point_at_angle_dist(
            p1=center_top,
            p2=shoulder_point,
            angle=sign * sampled_params["angle_between_arm_and_torso"],
            distance_from_end_point=-1 * sampled_params["shoulder_placement_ratio"] * np.linalg.norm(center_top - shoulder_point),
            new_line_distance=sampled_params["arm_length_to_torso_length_ratio"] * torso_length
        )
        
        self.vertex_mapping[f"{self.prefix}_elbow"] = elbow
        
        # Add outer and inner arm points
        for loc in ["outer", "inner"]:
            angle = 90 if (loc == "outer" and self.prefix == "left") or (loc == "inner" and self.prefix == "right") else -90
            
            # Proximal width
            central_upper_arm, outer_upper_arm = line_relative_to_end_point_at_angle_dist(
                p1=shoulder_point,
                p2=elbow,
                angle=angle,
                distance_from_end_point=-1 * sampled_params["proximal_arm_placement_ratio_above_elbow"] * np.linalg.norm(shoulder_point - elbow),
                new_line_distance=sampled_params["proximal_arm_width_to_torso_width_ratio"] * torso_width / 2
            )
            self.vertex_mapping[f"{self.prefix}_{loc}_upper_arm"] = outer_upper_arm
            
            # Distal width
            _, outer_lower_arm = line_relative_to_end_point_at_angle_dist(
                p1=shoulder_point,
                p2=elbow,
                angle=angle,
                distance_from_end_point=-1 * sampled_params["distal_arm_placement_ratio_above_elbow"] * np.linalg.norm(shoulder_point - elbow),
                new_line_distance=sampled_params["distal_arm_width_to_proximal_arm_width_ratio"] * np.linalg.norm(outer_upper_arm - central_upper_arm)
            )
            self.vertex_mapping[f"{self.prefix}_{loc}_lower_arm"] = outer_lower_arm
        
        return self.vertex_mapping

class LowerArm(DistalAppendage):
    def __init__(self, name: str = "lower_arm", prefix: str = None, parent = None, children = None, 
                create_params: dict[str, float] = {"forearm_length_to_upper_arm_length_ratio": (0.7, 0.9),
                                                   "proximal_forearm_width_to_elbow_width_ratio": (0.8, 1.0),
                                                   "distal_forearm_width_to_proximal_forearm_width_ratio": (0.6, 0.85),
                                                   "forearm_bulge_placement_ratio_above_wrist": (0.3, 0.5),
                                                   "proximal_forearm_placement_ratio_below_elbow": (0.0, 0.1),
                                                   "distal_forearm_placement_ratio_below_elbow": (0.9, 1.0),
                                                   "angle_between_forearm_and_upper_arm": (math.radians(-20), math.radians(20))},
                variation_factor: float = 0.05):
        super().__init__(
            name = name, 
            prefix = prefix, 
            parent = parent, 
            children = children,
            joint_name = "elbow",
            endpoint_name = "wrist",
            parent_joint_name = "shoulder",
            inner_point_name = "inner_lower_arm",
            outer_point_name = "outer_lower_arm",
            proximal_name = "proximal_forearm",
            distal_name = "distal_forearm",
            bulge_name = "forearm_bulge",
            create_params = create_params, 
            variation_factor = variation_factor
        )

class LowerLeg(DistalAppendage):
    def __init__(self, name: str = "lower_leg", prefix: str = None, parent = None, children = None, 
                create_params: dict[str, float] = {"lower_leg_length_to_thigh_length_ratio": (0.8, 1.1),
                                                   "proximal_lower_leg_width_to_knee_width_ratio": (0.8, 1.0),
                                                   "distal_lower_leg_width_to_proximal_lower_leg_width_ratio": (0.6, 0.8),
                                                   "calf_placement_ratio_above_ankle": (0.25, 0.4),
                                                   "proximal_lower_leg_placement_ratio_below_knee": (0.0, 0.1),
                                                   "distal_lower_leg_placement_ratio_below_knee": (0.9, 1.0),
                                                   "angle_between_lower_leg_and_thigh": (math.radians(-10), math.radians(10))},
                variation_factor: float = 0.05):
        super().__init__(
            name = name, 
            prefix = prefix, 
            parent = parent, 
            children = children,
            joint_name = "knee",
            endpoint_name = "ankle",
            parent_joint_name = "hip",
            inner_point_name = "inner_lower_thigh",
            outer_point_name = "outer_lower_thigh",
            proximal_name = "proximal_lower_leg",
            distal_name = "distal_lower_leg",
            bulge_name = "calf",
            create_params = create_params, 
            variation_factor = variation_factor
        )

if __name__ == "__main__":
    grid = generate_empty_grid()
    
    # Create the torso
    torso = Torso()
    
    # Create the head
    head = Head(parent=torso)
    
    # Create left and right upper arms with automatic parameter sharing and slight variations
    left_arm = UpperArm(prefix="left", parent=torso, variation_factor=0.05)
    right_arm = UpperArm(prefix="right", parent=torso, variation_factor=0.05)

    # Create left and right lower arms with automatic parameter sharing and slight variations
    left_lower_arm = LowerArm(prefix="left", parent=left_arm, variation_factor=0.05)
    right_lower_arm = LowerArm(prefix="right", parent=right_arm, variation_factor=0.05)

    # Create left and right thighs with automatic parameter sharing and slight variations
    left_thigh = Thigh(prefix="left", parent=torso, variation_factor=0.05)
    right_thigh = Thigh(prefix="right", parent=torso, variation_factor=0.05)
    
    # Create left and right lower legs with automatic parameter sharing and slight variations
    left_lower_leg = LowerLeg(prefix="left", parent=left_thigh, variation_factor=0.05)
    right_lower_leg = LowerLeg(prefix="right", parent=right_thigh, variation_factor=0.05)
    
    # Display the result
    plt.figure(figsize=(10, 8))
    
    # Function to compute bounding boxes for body parts
    def compute_bounding_boxes(body_parts, grid, custom_labels=None):
        from matplotlib.patches import Rectangle
        
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
        from matplotlib.patches import Rectangle
        
        for bbox in bboxes:
            if bbox:
                min_x, min_y, max_x, max_y = bbox['coords']
                box = Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, 
                                linewidth=2, edgecolor='b', facecolor='none')
                ax.add_patch(box)
                ax.text(min_x, min_y-5, f"{bbox['name']}: [{min_x},{min_y},{max_x},{max_y}]", 
                        color='blue', fontsize=8)
    
    # Command Pattern for Image Operations
    class ImageCommand:
        """Base class for image processing commands"""
        def execute(self, image):
            """Execute the command on the image"""
            pass
        
        def get_name(self):
            """Get the name of the command for display/labels"""
            return "Generic Command"
    
    class BlurCommand(ImageCommand):
        def __init__(self, sigma=1.0):
            self.sigma = sigma
        
        def execute(self, image):
            from scipy import ndimage
            return ndimage.gaussian_filter(image.astype(float), sigma=self.sigma)
        
        def get_name(self):
            return f"Blur (σ={self.sigma})"
    
    class ThresholdCommand(ImageCommand):
        def __init__(self, threshold_value=0.5):
            self.threshold_value = threshold_value
        
        def execute(self, image):
            return (image > self.threshold_value).astype(float)
        
        def get_name(self):
            return f"Threshold ({int(self.threshold_value*255)}/255)"
    
    class CommandChain:
        """Chain of commands to be applied in sequence"""
        def __init__(self, commands=None):
            self.commands = commands if commands else []
        
        def add_command(self, command):
            self.commands.append(command)
            return self
        
        def execute(self, image):
            result = image.copy()
            for command in self.commands:
                result = command.execute(result)
            return result
        
        def get_name(self):
            if not self.commands:
                return "Original"
            return " → ".join([cmd.get_name() for cmd in self.commands])
    
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
    
    # Plot body parts
    torso_mask = plot_body_part(torso, grid)
    head_mask = plot_body_part(head, grid)
    left_thigh_mask = plot_body_part(left_thigh, grid)
    right_thigh_mask = plot_body_part(right_thigh, grid)
    left_arm_mask = plot_body_part(left_arm, grid)
    right_arm_mask = plot_body_part(right_arm, grid)
    left_lower_arm_mask = plot_body_part(left_lower_arm, grid)
    right_lower_arm_mask = plot_body_part(right_lower_arm, grid)
    left_lower_leg_mask = plot_body_part(left_lower_leg, grid)
    right_lower_leg_mask = plot_body_part(right_lower_leg, grid)
    
    # Compute bounding boxes, keeping arms labeled separately
    all_bboxes = compute_bounding_boxes([
        torso,
        head,
        ([left_thigh, right_thigh], "proximal_legs"),  # Group thighs together
        ([left_lower_leg, right_lower_leg], "lower_legs"),  # Group lower legs together
        left_arm,
        right_arm,
        left_lower_arm, 
        right_lower_arm  # Group forearms together
    ], grid)
    
    # Draw bounding boxes on the main figure
    draw_bounding_boxes(plt.gca(), all_bboxes)
    
    # Combine masks
    combined_mask = np.maximum(torso_mask, head_mask)
    combined_mask = np.maximum(combined_mask, left_thigh_mask)
    combined_mask = np.maximum(combined_mask, right_thigh_mask)
    combined_mask = np.maximum(combined_mask, left_arm_mask)
    combined_mask = np.maximum(combined_mask, right_arm_mask)
    combined_mask = np.maximum(combined_mask, left_lower_arm_mask)
    combined_mask = np.maximum(combined_mask, right_lower_arm_mask)
    combined_mask = np.maximum(combined_mask, left_lower_leg_mask)
    combined_mask = np.maximum(combined_mask, right_lower_leg_mask)
    
    plt.imshow(combined_mask, cmap='gray')
    plt.title(f"Body Visualization")
    plt.colorbar()
    plt.show()
    
    # Use command pattern to create the requested visualizations
    
    # Create basic commands
    blur_weak = BlurCommand(sigma=.5)
    threshold_high = ThresholdCommand(threshold_value=0.97)
    
    # Create command chains
    original_chain = CommandChain()
    blur_weak_chain = CommandChain().add_command(blur_weak)
    
    # Create combined blur+threshold chains
    blur_thresh_low = CommandChain().add_command(blur_weak).add_command(threshold_high)

    
    # Visualize results
    # First show original and blurred variations
    visualize_command_results(
        combined_mask, 
        [original_chain, blur_weak_chain, blur_thresh_low], 
        all_bboxes, 
        title="Original and Blurred Variations"
    )
    