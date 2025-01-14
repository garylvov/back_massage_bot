#!/usr/bin/env bash
source /back_massage_bot/post-entry-hooks.sh && \
cd /ws/ && colcon build --symlink-install && \
source /ws/install/setup.bash || true
