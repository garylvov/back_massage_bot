#!/usr/bin/env bash
bash therapeutech/build.sh && \
docker build \
-t tt_ubuntu22_humble:latest \
--file ws/src/main_ros/Dockerfile \
--build-arg USERNAME="${USERNAME}" \
--progress=plain \
.