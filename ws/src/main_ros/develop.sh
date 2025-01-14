#!/usr/bin/env bash
xhost +local: || true  && \
 python3 docker.py -c bmb_ubuntu22_humble:latest -i -v back_massage_bot,ws/src/main_ros --devices "/dev/video0,/dev/video1,/dev/video2,/dev/video3,/dev/video4,/dev/video6,/dev/video7"
