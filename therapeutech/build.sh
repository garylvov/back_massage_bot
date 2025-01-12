#!/usr/bin/env bash
docker build \
-t tt_ubuntu22_python310:latest \
--build-arg UID=$(id -u) \
--build-arg GID=$(id -g) \
--build-arg USER=$USER \
--progress=plain \
--file therapeutech/Dockerfile \
.