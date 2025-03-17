#!/usr/bin/env bash
# xhost +local: || true  && \
#  python3 docker.py -c bmb_ubuntu22_python310:latest -i -v back_massage_bot

xhost +local: || true  && \
 python3 docker.py -c bmb_22_py10_backup:latest -i -v back_massage_bot
