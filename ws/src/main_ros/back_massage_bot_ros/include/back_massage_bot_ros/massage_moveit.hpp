// Copyright (c) 2025, Gary Lvov, Vinay Balaji, Tim Bennet, Xandar Ingare, Ben Yoon
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef WS_SRC_MAIN_ROS_BACK_MASSAGE_BOT_ROS_INCLUDE_BACK_MASSAGE_BOT_ROS_MASSAGE_MOVEIT_HPP_
#define WS_SRC_MAIN_ROS_BACK_MASSAGE_BOT_ROS_INCLUDE_BACK_MASSAGE_BOT_ROS_MASSAGE_MOVEIT_HPP_

// C system headers
// (none)

// C++ system headers
#include <memory>
#include <string>
#include <vector>

// Third-party headers (this comment helps maintain separation)
#include <geometry_msgs/msg/pose.hpp>
#include <rclcpp/rclcpp.hpp>

// MoveIt and TF2 headers (treated as project headers by cpplint)
#include "moveit/move_group_interface/move_group_interface.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace back_massage_bot_ros {

class MassageMoveit : public rclcpp::Node {
 public:
  explicit MassageMoveit(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  virtual ~MassageMoveit();

 private:
  // MoveIt interface
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  // TF2 components
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Parameters
  std::string arm_group_name_;
  std::string end_effector_link_;

  // Methods
  void initialize_move_group();
  bool move_to_pose(const geometry_msgs::msg::Pose& target_pose);
  bool move_to_joint_positions(const std::vector<double>& joint_positions);

  // Example massage patterns
  void execute_simple_massage_pattern();
};

}  // namespace back_massage_bot_ros

#endif  // WS_SRC_MAIN_ROS_BACK_MASSAGE_BOT_ROS_INCLUDE_BACK_MASSAGE_BOT_ROS_MASSAGE_MOVEIT_HPP_
