#!/usr/bin/env bash
xhost +local: || true  && \
 python3 docker.py tt_ubuntu22_humble:latest -i -v therapeutech,ws/src/main_ros