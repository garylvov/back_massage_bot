#!/usr/bin/env bash
cd /ws/src/main_ros
export PIXI_PROJECT_MANIFEST=/ws/src/main_ros/pixi.toml
pixi install
pixi run rosdep-init
pixi run rosdep-update
pixi run setup-ros
# Install the back_massage_bot package
cd /back_massage_bot && pip install -e .
cd /ws/src/main_ros
pixi run build-selected
pixi run setup-base-python

echo "ROS environment setup complete!"
pixi shell
