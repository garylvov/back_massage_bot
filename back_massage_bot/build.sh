#!/usr/bin/env bash
git submodule init && git submodule update && \
docker build \
-t bmb_ubuntu22_python310:latest \
--build-arg USERNAME="developer" \
--build-arg UID=1000 \
--build-arg GID=1000 \
--progress=plain \
--file back_massage_bot/Dockerfile \
.
