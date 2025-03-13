#!/usr/bin/env bash
git submodule init && git submodule update --recursive && \
bash back_massage_bot/build.sh && \
docker build \
-t bmb_ubuntu22_humble:latest \
--file ws/src/main_ros/Dockerfile \
--build-arg USERNAME="developer" \
--progress=plain \
.
