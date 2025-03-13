// Copyright (c) 2025, Gary Lvov, Vinay Balaji, Tim Bennet, Xandar Ingare, Ben Yoon
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include <chrono>  // NOLINT(build/c++11)
#include <thread>  // NOLINT(build/c++11)

#include "back_massage_bot_ros/massage_moveit.hpp"

namespace back_massage_bot_ros {

MassageMoveit::MassageMoveit(const rclcpp::NodeOptions& options) : Node("massage_moveit", options) {
  // Get parameters (don't declare them if they might be set from the launch file)
  if (!this->has_parameter("arm_group_name")) {
    this->declare_parameter("arm_group_name", "arm");
  }
  if (!this->has_parameter("end_effector_link")) {
    this->declare_parameter("end_effector_link", "j2n6s300_end_effector");
  }

  arm_group_name_ = this->get_parameter("arm_group_name").as_string();
  end_effector_link_ = this->get_parameter("end_effector_link").as_string();

  // Setup TF2
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Initialize MoveIt after a short delay to ensure ROS connections are established
  std::this_thread::sleep_for(std::chrono::seconds(2));
  initialize_move_group();

  // Execute a simple massage pattern as an example
  execute_simple_massage_pattern();
}

MassageMoveit::~MassageMoveit() {
  // Clean shutdown
  RCLCPP_INFO(this->get_logger(), "Shutting down massage_moveit node");
}

void MassageMoveit::initialize_move_group() {
  RCLCPP_INFO(this->get_logger(), "Initializing MoveGroup for '%s'", arm_group_name_.c_str());

  try {
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), arm_group_name_);

    // Set planning parameters
    move_group_->setMaxVelocityScalingFactor(0.1);  // Slow movements for safety
    move_group_->setMaxAccelerationScalingFactor(0.1);
    move_group_->setPlanningTime(5.0);
    move_group_->setNumPlanningAttempts(10);
    move_group_->setGoalPositionTolerance(0.01);     // 1cm
    move_group_->setGoalOrientationTolerance(0.01);  // ~0.6 degrees

    RCLCPP_INFO(this->get_logger(), "MoveGroup initialized successfully. Planning frame: %s, End effector: %s",
                move_group_->getPlanningFrame().c_str(), move_group_->getEndEffectorLink().c_str());
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to initialize MoveGroup: %s", e.what());
  }
}

bool MassageMoveit::move_to_pose(const geometry_msgs::msg::Pose& target_pose) {
  if (!move_group_) {
    RCLCPP_ERROR(this->get_logger(), "MoveGroup not initialized");
    return false;
  }

  RCLCPP_INFO(this->get_logger(), "Planning move to pose: [%f, %f, %f]", target_pose.position.x, target_pose.position.y,
              target_pose.position.z);

  move_group_->setPoseTarget(target_pose);

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool success = static_cast<bool>(move_group_->plan(plan));

  if (success) {
    RCLCPP_INFO(this->get_logger(), "Executing plan...");
    return static_cast<bool>(move_group_->execute(plan));
  } else {
    RCLCPP_ERROR(this->get_logger(), "Planning failed");
    return false;
  }
}

bool MassageMoveit::move_to_joint_positions(const std::vector<double>& joint_positions) {
  if (!move_group_) {
    RCLCPP_ERROR(this->get_logger(), "MoveGroup not initialized");
    return false;
  }

  RCLCPP_INFO(this->get_logger(), "Planning move to joint positions");

  move_group_->setJointValueTarget(joint_positions);

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool success = static_cast<bool>(move_group_->plan(plan));

  if (success) {
    RCLCPP_INFO(this->get_logger(), "Executing plan...");
    return static_cast<bool>(move_group_->execute(plan));
  } else {
    RCLCPP_ERROR(this->get_logger(), "Planning failed");
    return false;
  }
}

void MassageMoveit::execute_simple_massage_pattern() {
  RCLCPP_INFO(this->get_logger(), "Executing simple massage pattern");

  // First, move to a safe starting position
  // These are example joint positions - adjust for your specific robot
  std::vector<double> home_position = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  if (!move_to_joint_positions(home_position)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to move to home position");
    return;
  }

  // Define a simple massage pattern
  // This is just an example - you'll need to adjust these positions for your setup
  std::vector<geometry_msgs::msg::Pose> massage_points;

  // Create some example points
  for (int i = 0; i < 3; i++) {
    geometry_msgs::msg::Pose pose;

    // Base position
    pose.position.x = 0.4;  // Forward
    pose.position.y = 0.0;  // Center
    pose.position.z = 0.5;  // Height

    // Orientation - end effector pointing downward
    pose.orientation.x = 0.0;
    pose.orientation.y = 0.707;
    pose.orientation.z = 0.0;
    pose.orientation.w = 0.707;

    // Vary the y position for each point
    pose.position.y = -0.1 + (i * 0.1);

    massage_points.push_back(pose);
  }

  // Execute the pattern
  for (const auto& pose : massage_points) {
    if (!move_to_pose(pose)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to move to massage point");
      return;
    }

    // Pause at each point to simulate massage pressure
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  // Return to home position
  move_to_joint_positions(home_position);

  RCLCPP_INFO(this->get_logger(), "Massage pattern completed");
}

}  // namespace back_massage_bot_ros

// Main function
int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);

  auto node = std::make_shared<back_massage_bot_ros::MassageMoveit>(options);

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
