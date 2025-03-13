#!/usr/bin/env bash
xhost +local: || true  && \
docker pull garylvov/back_massage_bot:stable && \
docker build \
-t back_massage_bot_overlay \
--file Dockerfile \
--build-arg USERNAME="developer" \
--progress=plain \
. && \
python3 docker.py -c back_massage_bot_overlay -i -v back_massage_bot,ws/src/main_ros
