// Copyright (c) 2025, Gary Lvov, Tim Bennett, Xander Ingare, Ben Yoon, Vinay Balaji
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include <chrono>  // NOLINT(build/c++11)
#include <thread>  // NOLINT(build/c++11)
#include <memory>
#include <rclcpp/rclcpp.hpp>

#include "back_massage_bot_ros/massage_moveit.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"  // For doTransform

using namespace std::chrono_literals;

namespace back_massage_bot_ros {

MassageMoveit::MassageMoveit(const rclcpp::NodeOptions& options) : Node("massage_moveit", options) {
  try {
    // Get parameters (don't declare them if they might be set from the launch file)
    if (!this->has_parameter("arm_group_name")) {
      this->declare_parameter("arm_group_name", "arm");
    }
    if (!this->has_parameter("end_effector_link")) {
      this->declare_parameter("end_effector_link", "j2n6s300_end_effector");
    }
    if (!this->has_parameter("home_position")) {
      this->declare_parameter("home_position", std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    }

    // Planning parameters
    if (!this->has_parameter("velocity_scaling_factor")) {
      this->declare_parameter("velocity_scaling_factor", 0.1);
    }
    if (!this->has_parameter("acceleration_scaling_factor")) {
      this->declare_parameter("acceleration_scaling_factor", 0.1);
    }
    if (!this->has_parameter("planning_time")) {
      this->declare_parameter("planning_time", 5.0);
    }
    if (!this->has_parameter("planning_attempts")) {
      this->declare_parameter("planning_attempts", 10);
    }
    if (!this->has_parameter("goal_position_tolerance")) {
      this->declare_parameter("goal_position_tolerance", 0.01);
    }
    if (!this->has_parameter("goal_orientation_tolerance")) {
      this->declare_parameter("goal_orientation_tolerance", 0.01);
    }
    if (!this->has_parameter("planner_id")) {
      this->declare_parameter("planner_id", "RRTConnect");
    }
    if (!this->has_parameter("cartesian_path_eef_step")) {
      this->declare_parameter("cartesian_path_eef_step", 0.01);
    }
    if (!this->has_parameter("cartesian_path_jump_threshold")) {
      this->declare_parameter("cartesian_path_jump_threshold", 0.0);
    }

    // Get parameter values
    arm_group_name_ = this->get_parameter("arm_group_name").as_string();
    end_effector_link_ = this->get_parameter("end_effector_link").as_string();
    home_position_ = this->get_parameter("home_position").as_double_array();

    // Get planning parameter values
    velocity_scaling_factor_ = this->get_parameter("velocity_scaling_factor").as_double();
    acceleration_scaling_factor_ = this->get_parameter("acceleration_scaling_factor").as_double();
    planning_time_ = this->get_parameter("planning_time").as_double();
    planning_attempts_ = this->get_parameter("planning_attempts").as_int();
    goal_position_tolerance_ = this->get_parameter("goal_position_tolerance").as_double();
    goal_orientation_tolerance_ = this->get_parameter("goal_orientation_tolerance").as_double();
    planner_id_ = this->get_parameter("planner_id").as_string();
    cartesian_path_eef_step_ = this->get_parameter("cartesian_path_eef_step").as_double();
    cartesian_path_jump_threshold_ = this->get_parameter("cartesian_path_jump_threshold").as_double();

    // Setup TF2
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Setup subscriber for arm dispatch commands
    arm_dispatch_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/arm_dispatch_command", 10, std::bind(&MassageMoveit::arm_dispatch_callback, this, std::placeholders::_1));

    // Setup services
    return_home_service_ = this->create_service<std_srvs::srv::Trigger>(
        "/return_to_home",
        std::bind(&MassageMoveit::return_home_callback, this, std::placeholders::_1, std::placeholders::_2));

    stop_service_ = this->create_service<std_srvs::srv::Trigger>(
        "/stop_arm", std::bind(&MassageMoveit::stop_callback, this, std::placeholders::_1, std::placeholders::_2));

    // Schedule MoveGroup initialization after a delay
    // This ensures the node is fully initialized before we call shared_from_this()
    init_timer_ = this->create_wall_timer(std::chrono::seconds(15), std::bind(&MassageMoveit::delayed_init, this));

    RCLCPP_INFO(this->get_logger(), "MassageMoveit node initialized and ready");
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Exception in constructor: %s", e.what());
  } catch (...) {
    RCLCPP_ERROR(this->get_logger(), "Unknown exception in constructor");
  }
}

MassageMoveit::~MassageMoveit() {
  // Clean shutdown
  RCLCPP_INFO(this->get_logger(), "Shutting down massage_moveit node");
}

void MassageMoveit::delayed_init() {
  // This is called after the node is fully initialized
  RCLCPP_INFO(this->get_logger(), "Starting delayed initialization");

  try {
    // Cancel the timer so this only runs once
    init_timer_->cancel();

    // Now initialize the MoveGroup
    initialize_move_group();
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Exception in delayed_init: %s", e.what());
  } catch (...) {
    RCLCPP_ERROR(this->get_logger(), "Unknown exception in delayed_init");
  }
}

void MassageMoveit::initialize_move_group() {
  RCLCPP_INFO(this->get_logger(), "Initializing MoveGroup for '%s'", arm_group_name_.c_str());

  try {
    // Create the MoveGroupInterface with the correct constructor parameters
    RCLCPP_INFO(this->get_logger(), "Creating MoveGroupInterface");
    
    // Use the constructor that takes a node pointer, group name, and tf buffer
    // MoveGroupInterface(const rclcpp::Node::SharedPtr& node, const std::string& group,
    //                    const std::shared_ptr<tf2_ros::Buffer>&, const rclcpp::Duration&)
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(),  // Node pointer
      arm_group_name_,     // Group name
      tf_buffer_,          // TF buffer
      rclcpp::Duration::from_seconds(5.0)  // Wait timeout
    );
    
    RCLCPP_INFO(this->get_logger(), "MoveGroupInterface created successfully");

    // Set planning parameters from ROS parameters
    RCLCPP_INFO(this->get_logger(), "Setting planning parameters");
    
    move_group_->setMaxVelocityScalingFactor(velocity_scaling_factor_);
    move_group_->setMaxAccelerationScalingFactor(acceleration_scaling_factor_);
    move_group_->setPlanningTime(planning_time_);
    move_group_->setNumPlanningAttempts(planning_attempts_);
    move_group_->setGoalPositionTolerance(goal_position_tolerance_);
    move_group_->setGoalOrientationTolerance(goal_orientation_tolerance_);
    move_group_->setPlannerId(planner_id_);

    RCLCPP_INFO(this->get_logger(), "MoveGroup initialized successfully with the following parameters:");
    RCLCPP_INFO(this->get_logger(), "  Planning frame: %s", move_group_->getPlanningFrame().c_str());
    RCLCPP_INFO(this->get_logger(), "  End effector: %s", move_group_->getEndEffectorLink().c_str());
    RCLCPP_INFO(this->get_logger(), "  Velocity scaling factor: %.2f", velocity_scaling_factor_);
    RCLCPP_INFO(this->get_logger(), "  Acceleration scaling factor: %.2f", acceleration_scaling_factor_);
    RCLCPP_INFO(this->get_logger(), "  Planning time: %.2f", planning_time_);
    RCLCPP_INFO(this->get_logger(), "  Planning attempts: %d", planning_attempts_);
    RCLCPP_INFO(this->get_logger(), "  Goal position tolerance: %.3f", goal_position_tolerance_);
    RCLCPP_INFO(this->get_logger(), "  Goal orientation tolerance: %.3f", goal_orientation_tolerance_);
    RCLCPP_INFO(this->get_logger(), "  Planner ID: %s", planner_id_.c_str());
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to initialize MoveGroup: %s", e.what());
    RCLCPP_INFO(this->get_logger(),
                "This is normal if MoveIt is not running or if the robot description is not available.");
    RCLCPP_INFO(this->get_logger(), "The node will continue running, but arm control functionality will be disabled.");
    RCLCPP_INFO(this->get_logger(),
                "To use arm control, please launch MoveIt for your robot first, then restart this node.");
  } catch (...) {
    RCLCPP_ERROR(this->get_logger(), "Unknown exception during MoveGroup initialization");
    RCLCPP_INFO(this->get_logger(), "The node will continue running, but arm control functionality will be disabled.");
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

  // Use Cartesian path planning
  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.push_back(target_pose);

  moveit_msgs::msg::RobotTrajectory trajectory;

  // Try Cartesian path first
  double fraction = move_group_->computeCartesianPath(waypoints, cartesian_path_eef_step_,
                                                      cartesian_path_jump_threshold_, trajectory);

  if (fraction >= 0.9) {  // If we can achieve at least 90% of the path
    RCLCPP_INFO(this->get_logger(), "Cartesian path computed (%.2f%% achieved)", fraction * 100.0);
    return static_cast<bool>(move_group_->execute(trajectory));
  } else {
    RCLCPP_INFO(this->get_logger(), "Cartesian planning failed (%.2f%% achieved), trying regular planning",
                fraction * 100.0);

    // Fall back to regular planning
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

void MassageMoveit::arm_dispatch_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  RCLCPP_INFO(this->get_logger(), "Received arm dispatch command");

  if (!move_group_) {
    RCLCPP_ERROR(this->get_logger(), "MoveGroup not initialized yet. Cannot process command.");
    return;
  }

  // If the frame_id is not the planning frame, transform the pose
  geometry_msgs::msg::Pose target_pose = msg->pose;
  std::string planning_frame = move_group_->getPlanningFrame();

  if (msg->header.frame_id != planning_frame && !msg->header.frame_id.empty()) {
    try {
      geometry_msgs::msg::PoseStamped transformed_pose;
      transformed_pose.header.frame_id = planning_frame;
      transformed_pose.header.stamp = this->get_clock()->now();

      // Wait for transform to be available
      if (!tf_buffer_->canTransform(planning_frame, msg->header.frame_id, tf2::TimePointZero)) {
        RCLCPP_ERROR(this->get_logger(), "Cannot transform from %s to %s", msg->header.frame_id.c_str(),
                     planning_frame.c_str());
        return;
      }

      // Transform the pose
      tf_buffer_->transform(*msg, transformed_pose, planning_frame);
      target_pose = transformed_pose.pose;

    } catch (const tf2::TransformException& ex) {
      RCLCPP_ERROR(this->get_logger(), "Transform error: %s", ex.what());
      return;
    }
  }

  // Execute the movement
  if (!move_to_pose(target_pose)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to move to target pose");
  }
}

bool MassageMoveit::return_home_callback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                         std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
  // Silence unused parameter warning
  (void)request;

  RCLCPP_INFO(this->get_logger(), "Return to home service called");

  if (!move_group_) {
    response->success = false;
    response->message = "MoveGroup not initialized yet. Try again later.";
    return true;
  }

  if (home_position_.empty()) {
    response->success = false;
    response->message = "Home position not defined";
    return true;
  }

  bool success = move_to_joint_positions(home_position_);

  response->success = success;
  if (success) {
    response->message = "Successfully returned to home position";
  } else {
    response->message = "Failed to return to home position";
  }

  return true;
}

bool MassageMoveit::stop_callback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                  std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
  // Silence unused parameter warning
  (void)request;

  RCLCPP_INFO(this->get_logger(), "Stop arm service called");

  if (!move_group_) {
    response->success = false;
    response->message = "MoveGroup not initialized yet";
    return true;
  }

  move_group_->stop();

  response->success = true;
  response->message = "Arm movement stopped";

  return true;
}

}  // namespace back_massage_bot_ros

// Main function
int main(int argc, char **argv)
{
    // Initialize ROS
    rclcpp::init(argc, argv);
    
    // Create logger for main function
    auto logger = rclcpp::get_logger("massage_moveit_main");
    RCLCPP_INFO(logger, "Starting massage_moveit_node");
    
    try {
        // Create the node with a separate thread for the executor
        auto node = std::make_shared<back_massage_bot_ros::MassageMoveit>();
        
        // Create a multithreaded executor for better performance
        rclcpp::executors::MultiThreadedExecutor executor;
        executor.add_node(node);
        
        // Log that we're waiting before initializing MoveGroup
        RCLCPP_INFO(logger, "Waiting 5 before initializing MoveGroup to ensure all parameters are loaded...");
        
        // Create a separate thread for delayed initialization
        std::thread init_thread([node, &logger]() {
            // Sleep to allow time for MoveIt to fully initialize
            std::this_thread::sleep_for(std::chrono::seconds(5));
            
            try {
                RCLCPP_INFO(logger, "Attempting to initialize MoveGroup...");
                node->initialize_move_group();
                RCLCPP_INFO(logger, "MoveGroup initialization successful!");
            } catch (const std::exception& e) {
                RCLCPP_ERROR(logger, "Failed to initialize MoveGroup: %s", e.what());
                RCLCPP_WARN(logger, "Node will continue running but movement capabilities will be limited");
            }
        });
        
        // Detach the thread so it can run independently
        init_thread.detach();
        
        RCLCPP_INFO(logger, "Starting executor...");
        executor.spin();
        
        RCLCPP_INFO(logger, "Shutting down...");
        rclcpp::shutdown();
        return 0;
    } catch (const std::exception& e) {
        RCLCPP_FATAL(logger, "Unhandled exception in main: %s", e.what());
        rclcpp::shutdown();
        return 1;
    } catch (...) {
        RCLCPP_FATAL(logger, "Unknown exception in main");
        rclcpp::shutdown();
        return 1;
    }
}
