#!/usr/bin/env bash

# Source hooks
source post-entry-hooks.sh

# Launch processes in background and capture logs
ros2 launch back_massage_bot_ros esp32-bringup.py > esp32_log.txt &
ros2 launch back_massage_bot_ros massage-bringup.py > massage_log.txt &
ros2 launch back_massage_bot_ros camera-bringup.py > realsense_log.txt &
ros2 launch back_massage_bot_ros plan-massage-bringup.py > plan_massage_log.txt &
ros2 launch kinova_bringup kinova_robot_launch.py use_jaco_v1_fingers:=false > kinova_log.txt &
ros2 run kinova_driver joint_trajectory_action_server j2n6s300 > joint_trajectory_action_server_log.txt &
ros2 run kinova_driver gripper_command_action_server j2n6s300 > gripper_command_action_server_log.txt &
ros2 launch back_massage_bot_ros massage_moveit.launch.py 
