cmake_minimum_required(VERSION 3.8)
project(back_massage_bot_ros)

# Set compile options for better warnings and debugging
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)

# Include Cpp "include" directory
include_directories(include)

# Create Cpp executable
add_executable(massage_moveit_node src/massage_moveit_node.cpp)
ament_target_dependencies(
  massage_moveit_node
  rclcpp
  moveit_ros_planning_interface
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  std_srvs)

# Install Cpp executables
install(TARGETS massage_moveit_node DESTINATION lib/${PROJECT_NAME})

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS scripts/rgb_to_segmented_pose.py
        DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

# Enable testing and linters
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Skip lint checks temporarily (remove these lines when ready)
  set(AMENT_CMAKE_COPYRIGHT_FOUND TRUE)
  set(AMENT_CMAKE_CPPLINT_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
