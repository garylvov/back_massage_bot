#!/usr/bin/env bash
pushd . # Save current directory
source /back_massage_bot/post-entry-hooks.sh && \
cd /ws && \
colcon build \
--symlink-install \
--parallel-workers $(nproc) \
--packages-select \
back_massage_bot_ros \
synchros2 \
kinova_driver \
kinova_bringup \
kinova_msgs \
kinova_description \
kinova_demo \
--cmake-args \
-DCMAKE_LIBRARY_ARCHITECTURE=x86_64-linux-gnu \
-DPython3_EXECUTABLE=$HOME/miniforge3/envs/back_massage_bot/bin/python3.11
source /ws/install/setup.bash || true
popd # Return to original directory
