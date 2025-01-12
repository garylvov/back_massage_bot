#!/usr/bin/env bash
xhost +local: || true  && \
 python3 docker.py tt_ubuntu22_python310:latest -i -v therapeutech --x11
