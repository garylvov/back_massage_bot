#!/usr/bin/env bash
source /back_massage_bot/post-entry-hooks.sh && \
cd /ws/ && \
colcon build --symlink-install \
--cmake-args \
-DCMAKE_LIBRARY_ARCHITECTURE=x86_64-linux-gnu \
-DPython3_EXECUTABLE=$HOME/miniforge3/envs/back_massage_bot/bin/python3.11 && \
source /ws/install/setup.bash || true
