#!/usr/bin/env bash
xhost +local: || true  && \
 python3 docker.py tt_ubuntu22_humble:latest -i -v back_massage_bot,ws/src/main_ros
