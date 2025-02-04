#!/usr/bin/env bash
source post-entry-hooks.sh && \
ros2 launch back_massage_bot_ros camera-bringup.py > realsense_log.txt & \
ros2 launch kinova_bringup kinova_robot_launch.py > kinova_log.txt & \
ros2 launch kinova_bringup moveit_robot_launch.py > moveit_log.txt & \
ros2 run kinova_driver joint_trajectory_action_server j2n6s300 > joint_trajectory_action_server_log.txt & \
ros2 run kinova_driver gripper_command_action_server j2n6s300 > gripper_command_action_server_log.txt & 