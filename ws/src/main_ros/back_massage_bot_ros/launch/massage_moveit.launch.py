# Copyright (c) 2025, Gary Lvov, Vinay Balaji, Tim Bennet, Xandar Ingare, Ben Yoon
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Launch arguments for basic configuration
        DeclareLaunchArgument(
            "arm_group_name", default_value="arm", description="Name of the MoveIt group for the robot arm"
        ),
        DeclareLaunchArgument(
            "end_effector_link", default_value="j2n6s300_end_effector", description="Name of the end effector link"
        ),
        DeclareLaunchArgument(
            "home_position", default_value="[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]", description="Home position joint values"
        ),
        # Launch arguments for planning parameters
        DeclareLaunchArgument(
            "velocity_scaling_factor", default_value="0.1", description="Velocity scaling factor for planning (0.0-1.0)"
        ),
        DeclareLaunchArgument(
            "acceleration_scaling_factor",
            default_value="0.1",
            description="Acceleration scaling factor for planning (0.0-1.0)",
        ),
        DeclareLaunchArgument(
            "planning_time", default_value="5.0", description="Time allowed for motion planning (seconds)"
        ),
        DeclareLaunchArgument(
            "planning_attempts", default_value="10", description="Number of planning attempts before giving up"
        ),
        DeclareLaunchArgument(
            "goal_position_tolerance", default_value="0.01", description="Position tolerance for the goal (meters)"
        ),
        DeclareLaunchArgument(
            "goal_orientation_tolerance",
            default_value="0.01",
            description="Orientation tolerance for the goal (radians)",
        ),
        DeclareLaunchArgument(
            "planner_id", default_value="RRTConnect", description="Motion planner to use (e.g., RRTConnect, PRM, etc.)"
        ),
        DeclareLaunchArgument(
            "cartesian_path_eef_step",
            default_value="0.01",
            description="Step size for Cartesian path planning (meters)",
        ),
        DeclareLaunchArgument(
            "cartesian_path_jump_threshold",
            default_value="0.0",
            description="Jump threshold for Cartesian path planning",
        ),
        # Launch the massage_moveit node
        Node(
            package="back_massage_bot_ros",
            executable="massage_moveit_node",
            name="massage_moveit",
            output="screen",
            parameters=[{
                # Basic configuration
                "arm_group_name": LaunchConfiguration("arm_group_name"),
                "end_effector_link": LaunchConfiguration("end_effector_link"),
                "home_position": LaunchConfiguration("home_position"),
                # Planning parameters
                "velocity_scaling_factor": LaunchConfiguration("velocity_scaling_factor"),
                "acceleration_scaling_factor": LaunchConfiguration("acceleration_scaling_factor"),
                "planning_time": LaunchConfiguration("planning_time"),
                "planning_attempts": LaunchConfiguration("planning_attempts"),
                "goal_position_tolerance": LaunchConfiguration("goal_position_tolerance"),
                "goal_orientation_tolerance": LaunchConfiguration("goal_orientation_tolerance"),
                "planner_id": LaunchConfiguration("planner_id"),
                "cartesian_path_eef_step": LaunchConfiguration("cartesian_path_eef_step"),
                "cartesian_path_jump_threshold": LaunchConfiguration("cartesian_path_jump_threshold"),
            }],
        ),
    ])
