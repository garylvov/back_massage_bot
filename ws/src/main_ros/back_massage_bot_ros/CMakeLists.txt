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

# # Include Cpp "include" directory include_directories(include)

# # Create Cpp executable add_executable(cpp_executable src/cpp_node.cpp)
# ament_target_dependencies(cpp_executable rclcpp)

# # Install Cpp executables install(TARGETS cpp_executable DESTINATION
# lib/${PROJECT_NAME} )

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS scripts/rgb_to_segmented_pose.py
        DESTINATION lib/${PROJECT_NAME})

# Installmoveit2 controller
install(PROGRAMS scripts/arm_controller.py DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

# Enable testing and linters if(BUILD_TESTING) find_package(ament_lint_auto
# REQUIRED)

# # Skip lint checks temporarily (remove these lines when ready)
# set(ament_cmake_copyright_FOUND TRUE) set(ament_cmake_cpplint_FOUND TRUE)

# ament_lint_auto_find_test_dependencies() endif()

ament_package()
