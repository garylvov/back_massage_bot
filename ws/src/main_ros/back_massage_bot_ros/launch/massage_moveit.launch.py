# Copyright (c) 2025, Gary Lvov, Vinay Balaji, Tim Bennet, Xandar Ingare, Ben Yoon
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import os
import pathlib
import yaml
import xacro
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory, get_packages_with_prefixes


def generate_launch_description():
    # Get directories
    kinova_bringup_dir = get_package_share_directory('kinova_bringup')
    kinova_description_dir = get_package_share_directory('kinova_description')
    
    # Helper functions to load resources
    def load_file(package_dir, subpath):
        return pathlib.Path(os.path.join(package_dir, subpath)).read_text()

    def load_yaml(package_dir, subpath):
        return yaml.safe_load(load_file(package_dir, subpath))
    
    # Initialize the launch description nodes list
    launch_description_nodes = []
    
    # Check if moveit is installed
    if 'moveit' not in get_packages_with_prefixes():
        launch_description_nodes.append(LogInfo(msg='"moveit" package is not installed, \
                                                please install it in order to run this demo.'))
        return LaunchDescription(launch_description_nodes)
    
    # Load robot description
    xacro_file = os.path.join(kinova_description_dir, 'urdf', 'j2n6s300_standalone.xacro')
    doc = xacro.process_file(xacro_file)
    robot_description = {'robot_description': doc.toprettyxml(indent='  ')}
    
    # Load SRDF and other MoveIt configurations
    robot_description_semantic = {
        'robot_description_semantic': 
        load_file(kinova_bringup_dir, 'moveit_resource/j2n6s300.srdf')
    }
    
    kinematics_yaml = load_yaml(kinova_bringup_dir, 'moveit_resource/kinematics.yaml')
    robot_description_kinematics = {'robot_description_kinematics': kinematics_yaml}
    
    joint_limits_yaml = load_yaml(kinova_bringup_dir, 'moveit_resource/joint_limits.yaml')
    robot_description_planning = {'robot_description_planning': joint_limits_yaml}
    
    # Use the same controllers configuration as the Kinova MoveIt launch
    moveit_controllers = {
        'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
        'moveit_simple_controller_manager': 
        load_yaml(kinova_bringup_dir, 'moveit_resource/controllers.yaml')
    }
    
    # Planning Configuration
    ompl_planning_pipeline_config = {
        "move_group": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": """default_planner_request_adapters/AddTimeOptimalParameterization 
                                  default_planner_request_adapters/FixWorkspaceBounds 
                                  default_planner_request_adapters/FixStartStateBounds 
                                  default_planner_request_adapters/FixStartStateCollision 
                                  default_planner_request_adapters/FixStartStatePathConstraints""",
            "start_state_max_bounds_error": 0.1,
        }
    }
    
    # Update with OMPL planning configuration
    ompl_planning_yaml = load_yaml(kinova_bringup_dir, 'moveit_resource/ompl_planning.yaml')
    ompl_planning_pipeline_config["move_group"].update(ompl_planning_yaml)
    
    # Simulation time parameter
    sim_time = {'use_sim_time': False}
    
    # Add RViz node from Kinova launch
    rviz_config_file = os.path.join(kinova_bringup_dir, 'moveit_resource', 'visualization.rviz')
    
    launch_description_nodes.append(
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            parameters=[
                robot_description,
                robot_description_semantic,
                robot_description_kinematics,
                robot_description_planning,
                sim_time
            ],
            remappings=[('/joint_states', '/j2n6s300_driver/out/joint_state')],
        )
    )
    
    # Add MoveIt2 node from Kinova launch
    launch_description_nodes.append(
        Node(
            package='moveit_ros_move_group',
            executable='move_group',
            output='screen',
            parameters=[
                robot_description,
                robot_description_semantic,
                robot_description_kinematics,
                moveit_controllers,
                ompl_planning_pipeline_config,
                robot_description_planning,
                sim_time
            ],
            remappings=[('/joint_states', '/j2n6s300_driver/out/joint_state')],
        )
    )

    # Create the massage_moveit node with a delay to ensure MoveIt is fully initialized
    massage_moveit_node = Node(
        package="back_massage_bot_ros",
        executable="massage_moveit_node",
        name="massage_moveit",
        output="screen",
        parameters=[
            # Robot description parameters
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            robot_description_planning,
            moveit_controllers,
            ompl_planning_pipeline_config,
            sim_time,
            # Node-specific parameters
            {
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
            }
        ],
        remappings=[('/joint_states', '/j2n6s300_driver/out/joint_state')],
    )
    
    # Delay the start of the massage_moveit node to ensure MoveIt is fully initialized
    delayed_massage_moveit_node = TimerAction(
        period=5.0,  # 5     seconds delay
        actions=[massage_moveit_node]
    )
    
    # Add launch arguments and delayed massage_moveit node
    return LaunchDescription(
        launch_description_nodes + [
            # Add a log message to indicate we're waiting for MoveIt to start
            LogInfo(msg="Waiting for MoveIt to start..."),
            
            # Launch arguments for basic configuration
            DeclareLaunchArgument(
                "arm_group_name", default_value="arm", description="Name of the MoveIt group for the robot arm"
            ),
            DeclareLaunchArgument(
                "end_effector_link", default_value="j2n6s300_end_effector", description="Name of the end effector link"
            ),
            DeclareLaunchArgument(
                "home_position", default_value="[-0.175, 3.055, 1.991, -1.344, 3.071, 1.536]", description="Home position joint values"
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
                "goal_position_tolerance", default_value="0.0025", description="Position tolerance for the goal (meters)"
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
                default_value="0.001",
                description="Step size for Cartesian path planning (meters)",
            ),
            DeclareLaunchArgument(
                "cartesian_path_jump_threshold",
                default_value="0.0",
                description="Jump threshold for Cartesian path planning",
            ),
            
            # Launch the massage_moveit node with a delay
            delayed_massage_moveit_node,
        ]
    )
