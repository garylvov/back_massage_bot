#!/usr/bin/env bash
docker build \
-t tt_ubuntu22_humble:latest \
--file ws/src/main_ros/Dockerfile \
.