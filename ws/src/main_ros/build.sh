#!/usr/bin/env bash
bash back_massage_bot/build.sh && \
docker build \
-t bmb_ubuntu22_humble:latest \
--file ws/src/main_ros/Dockerfile \
--build-arg USERNAME="${USERNAME}" \
--progress=plain \
.
