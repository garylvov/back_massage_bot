#!/usr/bin/env python3
# Copyright (c) 2025, Gary Lvov, Tim Bennett, Xander Ingare, Ben Yoon, Vinay Balaji
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from back_massage_bot.synthetic_data_gen_util import *
import math
import os
import argparse
from PIL import Image
from tqdm import tqdm
import uuid

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

    @classmethod
    def reset_instances(cls):
        """Reset the global instances list to create a fresh body"""
        cls.instances = []

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
                create_params: dict[str, float] = {"torso_length": (0.3, 0.9), 
                                                    "shoulder_from_bottom_ratio": (0.85, 1.0),
                                                    "shoulder_width_torso_length_ratio": (0.25, 0.9),
                                                    "waist_from_bottom_ratio": (0.0, .05),
                                                    "waist_width_shoulder_width_ratio": (.3, 1.0)}):
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
                                                   "arm_length_to_torso_length_ratio": (0.25, 0.75),
                                                   "proximal_arm_width_to_torso_width_ratio": (0.15, 0.25),
                                                   "distal_arm_width_to_proximal_arm_width_ratio": (0.75, 0.95),
                                                   "proximal_arm_placement_ratio_above_elbow": (0.9, 1.0),
                                                   "distal_arm_placement_ratio_above_elbow": (0.0, 0.1),
                                                   "angle_between_arm_and_torso": (math.radians(-70), math.radians(70))},
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
                                                   "angle_between_forearm_and_upper_arm": (math.radians(-70), math.radians(70))},
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
                
def generate_random_body_parts(probs=None):
    """
    Generate random body parts based on probability parameters.
    
    Args:
        probs: Dictionary of probabilities for different body parts
        
    Returns:
        Tuple of (body_parts, grid)
    """
    if probs is None:
        probs = {
            "torso": 0.99,
            "head": 0.95,
            "left_arm": 0.95,
            "right_arm": 0.95,
            "left_leg": 0.95,
            "right_leg": 0.95,
        }
    
    grid = np.ones((200, 200))  # Placeholder grid
    body_parts = []
    
    # Create torso if probability check passes
    if np.random.random() < probs["torso"]:
        torso = Torso()
        body_parts.append(torso)
        
        # Create head if probability check passes and torso exists
        if np.random.random() < probs["head"]:
            head = Head(parent=torso)
            body_parts.append(head)
        
        # Create arms if probability checks pass and torso exists
        for side, prob_key in [("left", "left_arm"), ("right", "right_arm")]:
            if np.random.random() < probs[prob_key]:
                upper_arm = UpperArm(prefix=side, parent=torso)
                body_parts.append(upper_arm)
                
                # Lower arm is conditional on upper arm
                lower_arm = LowerArm(prefix=side, parent=upper_arm)
                body_parts.append(lower_arm)
        
        # Create legs if probability checks pass and torso exists
        for side, prob_key in [("left", "left_leg"), ("right", "right_leg")]:
            if np.random.random() < probs[prob_key]:
                leg = Thigh(prefix=side, parent=torso)
                body_parts.append(leg)
                
                # Lower leg is conditional on upper leg/thigh
                lower_leg = LowerLeg(prefix=side, parent=leg)
                body_parts.append(lower_leg)
    
    return body_parts, grid


if __name__ == "__main__":
    # Add command-line arguments
    parser = argparse.ArgumentParser(description='Generate synthetic body part data')
    parser.add_argument('--dir', type=str, help='Directory to save generated data')
    parser.add_argument('--viz', action='store_true', help='Visualize bounding boxes')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
    args = parser.parse_args()
    
    # Create output directories if specified
    if args.dir:
        os.makedirs(os.path.join(args.dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(args.dir, 'labels'), exist_ok=True)
    
    # Generate specified number of samples
    for i in tqdm(range(args.num_samples), desc="Generating samples"):
        # Reset body parts to ensure independent generation for each sample
        BodyPart.reset_instances()
        
        # Generate unique ID for this sample
        sample_id = str(uuid.uuid4())[:8]
        
        # Create an empty grid
        grid_shape = (200, 200)
        grid = np.zeros(grid_shape)
        
        # Generate random body parts
        body_parts, grid = generate_random_body_parts()
        
        if not body_parts:
            # Empty image case
            combined_mask = np.zeros(grid.shape, dtype=bool)
            
            if args.viz:
                plt.figure(figsize=(10, 8))
                plt.imshow(combined_mask, cmap='gray')
                plt.title("Empty Body Visualization")
                plt.colorbar()
                plt.show()
        else:
            # Plot body parts and create masks
            combined_mask = np.zeros(grid.shape, dtype=bool)
            
            # Define grouped parts for bbox computation
            bbox_groups = []
            leg_parts = []  # Collect all leg parts (both thighs and lower legs)
            
            # Define a local function that gets the mask without plotting vertices
            def get_part_mask(part, grid):
                vertices = part.get_vertices()
                try:
                    return create_filled_curve_mask(grid.shape, points=vertices, curves_enabled=True, curve_tension=0.1)
                except Exception as e:
                    print(f"Error creating mask for {part.name}: {e}")
                    return np.zeros(grid.shape, dtype=bool)
            
            for part in body_parts:
                # Use our local function that doesn't plot vertices
                part_mask = get_part_mask(part, grid)
                combined_mask = np.maximum(combined_mask, part_mask)
                
                # Organize parts for bounding box groups
                if isinstance(part, Thigh) or isinstance(part, LowerLeg):
                    leg_parts.append(part)
                elif not isinstance(part, BodyPart):
                    continue
                else:
                    bbox_groups.append(part)
            
            # Add all leg parts as a single group if they exist
            if leg_parts:
                bbox_groups.append((leg_parts, "legs"))
            
            # Compute bounding boxes
            all_bboxes = []
            if body_parts:  # Only generate bounding boxes if there are actual body parts
                all_bboxes = compute_bounding_boxes(bbox_groups, grid)
                all_bboxes = process_and_filter_bboxes(all_bboxes, combined_mask, grid)
            
            # Create binary-only noise effects 
            # 1. Clusters removing from body (foreground)
            body_clusters = ClusteredNoiseCommand(
                density=(0.05, 0.12),     # Higher density 
                cluster_size=(3, 10),     # Size of clusters
                value=0.0,                # Value to set (black)
                target_value=1.0,         # Target white areas (the body)
                border_focus=0.3,         # Focus some clusters near borders
                probability=0.8           # 80% chance of applying
            )

            # 2. Clusters adding to background
            bg_clusters = ClusteredNoiseCommand(
                density=(0.02, 0.06),     # Background density
                cluster_size=(2, 8),      # Size of clusters
                value=1.0,                # Value to set (white) 
                target_value=0.0,         # Target black areas (background)
                border_focus=0.0,         # Uniform distribution
                probability=0.9           # 90% chance of applying
            )

            # 3. Lines on body (removing)
            body_lines = RandomLinesCommand(
                num_lines=(8, 15),        # Number of lines
                thickness=(1, 3),         # Line thickness
                length=(15, 40),          # Line length
                value=0.0,                # Value to set (black)
                target_value=1.0,         # Target white areas (body)
                probability=0.7           # 70% chance of applying
            )

            # 4. Lines on background (adding)
            bg_lines = RandomLinesCommand(
                num_lines=(5, 12),        # Number of lines
                thickness=(1, 2),         # Line thickness  
                length=(10, 30),          # Line length
                value=1.0,                # Value to set (white)
                target_value=0.0,         # Target black areas (background)
                probability=0.8           # 80% chance of applying
            )

            # Create command chain
            command_chain = CommandChain()
            command_chain.add_command(BlurCommand(sigma=(0.5, 0.8), probability=0.9))  # Initial blur

            # Add the new pixel flip command to get ~30% flipped pixels
            # You can target foreground, background, or both
            command_chain.add_command(RandomPixelFlipCommand(
                flip_percentage=(0.00, 0.08),  # Target around 30% flipped pixels
                target_value=None,             # Apply to both foreground and background
                probability=1.0                # Always apply
            ))

            # Optionally keep some of your existing noise commands for texture variation
            if np.random.random() < 0.5:  # 50% chance to add additional noise
                command_chain.add_command(body_clusters)
                command_chain.add_command(body_lines)
                command_chain.add_command(bg_clusters)

            # Add thresholding to ensure binary output  
            command_chain.add_command(ThresholdCommand(threshold_value=(0.4, 0.6), probability=1.0))

            # Apply the processing chain
            processed_mask = command_chain.execute(combined_mask)
            
            # Visualize if requested
            if args.viz:
                # Create a single figure for the final processed mask with bounding boxes
                plt.figure(figsize=(10, 8))
                plt.imshow(processed_mask, cmap='gray')
                plt.title("Body Part Mask with Bounding Boxes")
                
                # Draw bounding boxes on the current axes
                ax = plt.gca()
                draw_bounding_boxes(ax, all_bboxes)
                
                plt.colorbar()
                plt.tight_layout()
                plt.show()
            
            # Save to files if directory provided
            if args.dir:
                # Convert processed_mask to uint8 image (0-255) with proper contrast
                if processed_mask.dtype == bool:
                    # Convert boolean to integer then to uint8 with full range
                    img_array = (processed_mask * 255).astype(np.uint8)
                else:
                    # Normalize to 0-255 range for better visibility
                    min_val = np.min(processed_mask)
                    max_val = np.max(processed_mask)
                    if max_val > min_val:  # Avoid division by zero
                        img_array = (((processed_mask - min_val) / (max_val - min_val)) * 255).astype(np.uint8)
                    else:
                        img_array = np.zeros_like(processed_mask, dtype=np.uint8)
                
                # Create a Pillow image with proper mode
                img = Image.fromarray(img_array, mode='L')  # 'L' is for grayscale
                
                # Save image in images subdirectory
                img_path = os.path.join(args.dir, 'images', f'{sample_id}.png')
                img.save(img_path)
            
            # Save labels if there are bounding boxes
            if all_bboxes:  # Check that we have bounding boxes
                label_path = os.path.join(args.dir, 'labels', f'{sample_id}.txt')
                save_yolo_labels(all_bboxes, label_path, grid.shape)
            else:
                # Create empty label file if directory provided (important for training)
                if args.dir:
                    label_path = os.path.join(args.dir, 'labels', f'{sample_id}.txt')
                    # Create an empty file by opening and closing it
                    with open(label_path, 'w') as f:
                        pass  # Empty file = no labels
    