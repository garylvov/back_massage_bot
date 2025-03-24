#!/usr/bin/env python3
# Copyright (c) 2025, Gary Lvov, Tim Bennett, Xander Ingare, Ben Yoon, Vinay Balaji
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from back_massage_bot.synthetic_data_gen_util import *
from back_massage_bot.augmentations import *
import math
import os
import argparse
from PIL import Image
from tqdm import tqdm
import uuid
import multiprocessing as mp

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
                create_params: dict[str, float] = {"head_length_to_torso_ratio": (0.15, 0.45),
                                                  "head_width_to_length_ratio": (0.7, 1.0),
                                                  "head_vertical_offset_ratio": (-0.05, 0.1)}):
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
        
        # Calculate head center
        head_center = head_bottom + np.array([0, head_length/2])
        
        # Define more points around the elliptical shape to make it smoother
        # Use parametric equation of ellipse: x = a*cos(t), y = b*sin(t)
        # where a = width/2, b = length/2
        a = head_width / 2
        b = head_length / 2
        
        # Add points at various angles around the ellipse
        angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        for i, angle in enumerate(angles):
            theta = math.radians(angle)
            # Convert to cartesian coordinates
            x = a * math.cos(theta)
            y = b * math.sin(theta)
            
            # Position relative to head center
            point = head_center + np.array([x, y])
            
            # Name based on position around ellipse
            if angle == 0:
                point_name = "right_head"
            elif angle == 90:
                point_name = "top_head"
            elif angle == 180:
                point_name = "left_head"
            elif angle == 270:
                point_name = "bottom_head"
            else:
                # Position descriptions for intermediate points
                quadrant = angle // 90
                if quadrant == 0:
                    point_name = f"top_right_head_{angle}"
                elif quadrant == 1:
                    point_name = f"top_left_head_{angle}"
                elif quadrant == 2:
                    point_name = f"bottom_left_head_{angle}"
                else:
                    point_name = f"bottom_right_head_{angle}"
                
            self.vertex_mapping[point_name] = point
        
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
            "head": 1.0,
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

def adjust_torso_bounding_box(all_bboxes, shrink_bottom_percent=0.25, shrink_top_percent=0.1):
    """
    Adjusts the torso bounding box by shrinking it from the top and bottom
    
    Args:
        all_bboxes (list): List of bounding box dictionaries
        shrink_bottom_percent (float): Percentage to shrink from the bottom (0.5 = 50%)
        shrink_top_percent (float): Percentage to shrink from the top (0.5 = 50%)
        
    Returns:
        list: Updated bounding boxes
    """
    torso_found = False
    for bbox in all_bboxes:
        # Case-insensitive check for "torso" in the name
        if 'name' in bbox and bbox['name'].lower() == 'torso':
            torso_found = True
            
            # Check for different possible key formats
            if 'coords' in bbox and isinstance(bbox['coords'], tuple) and len(bbox['coords']) == 4:
                # Format is (x_min, y_min, x_max, y_max)
                x_min, y_min, x_max, y_max = bbox['coords']
                height = y_max - y_min
                
                # Calculate new boundaries
                bottom_adjustment = height * shrink_bottom_percent
                top_adjustment = height * shrink_top_percent
                
                # Store original values for debugging
                original_coords = bbox['coords']
                
                # Create new tuple with adjusted values (keeping x coordinates the same)
                bbox['coords'] = (x_min, y_min + top_adjustment, x_max, y_max - bottom_adjustment)
            elif 'ymin' in bbox and 'ymax' in bbox:
                # Get original dimensions
                y_min, y_max = bbox['ymin'], bbox['ymax']
                height = y_max - y_min
                
                # Calculate new boundaries
                bottom_adjustment = height * shrink_bottom_percent
                top_adjustment = height * shrink_top_percent
                
                # Update the bounding box
                bbox['ymin'] = y_min + top_adjustment  # Shrink from top
                bbox['ymax'] = y_max - bottom_adjustment  # Shrink from bottom
            elif 'min_y' in bbox and 'max_y' in bbox:
                # Get original dimensions
                y_min, y_max = bbox['min_y'], bbox['max_y']
                height = y_max - y_min
                
                # Calculate new boundaries
                bottom_adjustment = height * shrink_bottom_percent
                top_adjustment = height * shrink_top_percent
                
                # Update the bounding box
                bbox['min_y'] = y_min + top_adjustment  # Shrink from top
                bbox['max_y'] = y_max - bottom_adjustment  # Shrink from bottom
    
    if not torso_found:
        print("No torso found in bounding boxes!")
        if all_bboxes:
            print(f"Available bounding box names: {[bbox.get('name', 'unnamed') for bbox in all_bboxes]}")
    
    return all_bboxes

def generate_sample(i, args, return_dict=None):
    """Generate a single sample - isolates the body part instances"""
    # Reset body parts to ensure independent generation
    BodyPart.reset_instances()
    
    # Generate unique ID for this sample
    sample_id = str(uuid.uuid4())[:8]
    
    # Create an empty grid
    grid_shape = (200, 200)
    grid = np.zeros(grid_shape)
    
    # Generate random body parts
    body_parts, grid = generate_random_body_parts()
    
    # Validate body parts and create mask
    is_valid, combined_mask, bbox_groups = validate_body_parts(body_parts, grid, min_body_pixels=1000)
    
    if not is_valid:
        # Skip invalid samples
        return False
    
    # Compute bounding boxes
    all_bboxes = compute_bounding_boxes(bbox_groups, grid)

    # AGGRESSIVE ARM FILTERING: Remove nested arms BEFORE any other checks
    all_bboxes = filter_nested_arms(all_bboxes, max_overlap_ratio=0.3)
    
    # Only fine-tune the torso bounding box if the flag is enabled
    if args.fine_tune_torso:
        all_bboxes = adjust_torso_bounding_box(all_bboxes)
    
    # Skip if no valid bounding boxes after filtering
    if not all_bboxes:
        return False
    
    # Store original mask for comparison
    original_mask = combined_mask.copy()
    
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

    bg_cluster_max = 8
    if args.fine_tune_torso:
        bg_cluster_max = 4
    # 2. Clusters adding to background
    bg_clusters = ClusteredNoiseCommand(
        density=(0.02, 0.06),     # Background density
        cluster_size=(2, bg_cluster_max),      # Size of clusters
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

    # Create command chain
    command_chain = CommandChain()
    command_chain.add_command(BlurCommand(sigma=(0.5, 0.8), probability=0.9))  # Initial blur

    # Add the pixel flip command
    command_chain.add_command(RandomPixelFlipCommand(
        flip_percentage=(0.00, 0.08),  # Target around 8% flipped pixels
        target_value=None,             # Apply to both foreground and background
        probability=1.0                # Always apply
    ))

    # Optionally keep some noise commands for texture variation
    if np.random.random() < 0.5:  # 50% chance to add additional noise
        command_chain.add_command(body_clusters)
        command_chain.add_command(body_lines)
        command_chain.add_command(bg_clusters)

    # Add thresholding to ensure binary output  
    command_chain.add_command(ThresholdCommand(threshold_value=(0.4, 0.6), probability=1.0))

    # Apply the processing chain
    processed_mask = command_chain.execute(combined_mask)
    
    # Validate the processed mask
    if not validate_processed_mask(original_mask, processed_mask, min_body_pixels=2000, retention_ratio=0.7):
        return False

    # Validate that each bounding box contains sufficient pixels
    if not perform_final_validation(processed_mask, all_bboxes, min_body_pixels=2000, min_density=0.15, min_pixels=100):
        return False

    # Additional visual sanity check
    box_contents = []
    for box in all_bboxes:
        name = box.get('name', 'unknown')
        density, foreground_pixels, total_pixels = calculate_box_density(box, processed_mask)
        box_contents.append((name, foreground_pixels, density))

    # Skip if any box has low density or few pixels
    if any(density < 0.03 or pixels < 50 for _, pixels, density in box_contents):
        return False
    
    # Visualize if requested
    if args.viz:
        # Create a single figure for visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(processed_mask, cmap='gray')
        plt.title(f"Sample {i+1}: Body Part Mask with Bounding Boxes")
        
        # Draw bounding boxes
        ax = plt.gca()
        draw_bounding_boxes(ax, all_bboxes)
        
        plt.colorbar()
        plt.tight_layout()
        plt.show()
    
    # Save to files if directory provided
    if args.dir:
        # Convert to image
        img_array = (processed_mask * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L')
        
        # Save image
        img_path = os.path.join(args.dir, 'images', f'{sample_id}.png')
        img.save(img_path)
    
        # Save labels
        if all_bboxes:
            label_path = os.path.join(args.dir, 'labels', f'{sample_id}.txt')
            save_yolo_labels(all_bboxes, label_path, grid.shape)
    
    # Store results in shared dictionary if using multiprocessing
    if return_dict is not None:
        return_dict[i] = True
    
    return True

def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic data for back massage robot')
    parser.add_argument('--dir', type=str, help='Directory to save generated data')
    parser.add_argument('--viz', action='store_true', help='Visualize bounding boxes')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel workers for generation')
    parser.add_argument('--fine-tune-torso', action='store_true', 
                        help='Fine-tune torso bounding box by shrinking 30% from bottom and 20% from top')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Add command-line arguments
    args = parse_args()
    
    # Create output directories if specified
    if args.dir:
        os.makedirs(os.path.join(args.dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(args.dir, 'labels'), exist_ok=True)
    
    # Initialize progress tracking
    total_generated = 0
    target_samples = args.num_samples
    
    # Visualization mode doesn't work well with parallel processing
    if args.viz:
        args.num_workers = 1
    
    # Initialize multiprocessing if using multiple workers
    if args.num_workers > 1:
        # Use a Manager to track progress across processes
        with mp.Manager() as manager:
            return_dict = manager.dict()
            
            # Create a pool of workers
            with mp.Pool(processes=args.num_workers) as pool:
                # Generate samples until we reach the target
                while total_generated < target_samples:
                    # Determine how many more samples to try generating
                    samples_to_generate = min(args.num_workers * 2, target_samples - total_generated + 10)
                    
                    # Start parallel generation
                    results = [pool.apply_async(generate_sample, args=(i, args, return_dict)) 
                              for i in range(samples_to_generate)]
                    
                    # Wait for all processes to complete
                    for result in results:
                        result.get()
                    
                    # Count successful generations
                    successful = sum(1 for x in return_dict.values() if x)
                    total_generated += successful
                    
                    print(f"Generated {total_generated}/{target_samples} samples")
                    
                    # Clear the return dictionary for the next batch
                    return_dict.clear()
                    
                    # Exit if we've generated enough samples
                    if total_generated >= target_samples:
                        break
    else:
        # Single process mode
        pbar = tqdm(total=target_samples, desc="Generating samples")
        
        # Keep generating until we reach the target
        attempts = 0
        while total_generated < target_samples:
            if generate_sample(attempts, args):
                total_generated += 1
                pbar.update(1)
            attempts += 1
    