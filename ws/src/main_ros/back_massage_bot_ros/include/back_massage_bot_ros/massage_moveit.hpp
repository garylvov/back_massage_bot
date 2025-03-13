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
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

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

  // ROS communication
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr arm_dispatch_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr return_home_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stop_service_;
  rclcpp::TimerBase::SharedPtr init_timer_;  // Timer for delayed initialization

  // Parameters
  std::string arm_group_name_;
  std::string end_effector_link_;
  std::vector<double> home_position_;

  // Planning parameters
  double velocity_scaling_factor_;
  double acceleration_scaling_factor_;
  double planning_time_;
  int planning_attempts_;
  double goal_position_tolerance_;
  double goal_orientation_tolerance_;
  std::string planner_id_;
  double cartesian_path_eef_step_;
  double cartesian_path_jump_threshold_;

  // Methods
  void delayed_init();  // New method for delayed initialization
  void initialize_move_group();
  bool move_to_pose(const geometry_msgs::msg::Pose& target_pose);
  bool move_to_joint_positions(const std::vector<double>& joint_positions);

  // Callback methods
  void arm_dispatch_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  bool return_home_callback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                            std::shared_ptr<std_srvs::srv::Trigger::Response> response);
  bool stop_callback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                     std::shared_ptr<std_srvs::srv::Trigger::Response> response);
};

}  // namespace back_massage_bot_ros

#endif  // WS_SRC_MAIN_ROS_BACK_MASSAGE_BOT_ROS_INCLUDE_BACK_MASSAGE_BOT_ROS_MASSAGE_MOVEIT_HPP_
