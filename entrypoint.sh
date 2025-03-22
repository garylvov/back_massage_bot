#!/usr/bin/env bash
xhost +local: || true  && \
docker pull garylvov/back_massage_bot:stable && \
python3 docker.py -c garylvov/back_massage_bot:stable -e