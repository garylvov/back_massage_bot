#!/usr/bin/env bash
git submodule init && git submodule update && \
docker build \
-t bmb_ubuntu22_python310:latest \
--build-arg USERNAME="${USERNAME}" \
--build-arg UID=$(id -u) \
--build-arg GID=$(id -g) \
--build-arg USER=$USER \
--progress=plain \
--file back_massage_bot/Dockerfile \
.
