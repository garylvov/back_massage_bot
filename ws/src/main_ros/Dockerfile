FROM bmb_ubuntu22_python310:latest

# Define username variable
ARG USERNAME=garylvov

USER root
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libglvnd0 \
    libglx0 \
    libopengl0 \
    libglx-mesa0 \
    libglu1-mesa \
    libglu1-mesa-dev \
    x11-xserver-utils

# Create workspace directories
RUN mkdir -p /ws/src && chown -R ${USERNAME}:${USERNAME} /ws

# Switch to user
USER ${USERNAME}
WORKDIR /ws/src/main_ros

# Copy pixi configuration first
COPY --chown=${USERNAME}:${USERNAME} ws/src/main_ros/pixi.toml /ws/src/main_ros/pixi.toml

# Install pixi environment
RUN cd /ws/src/main_ros && pixi install

# Copy the rest of the ROS code
COPY --chown=${USERNAME}:${USERNAME} ws/src/main_ros /ws/src/main_ros/

# Create post-entry hooks
COPY --chown=${USERNAME}:${USERNAME} back_massage_bot/post-entry-hooks.sh /back_massage_bot/post-entry-hooks.sh
COPY --chown=${USERNAME}:${USERNAME} ws/src/main_ros/post-entry-hooks.sh /ws/src/main_ros/post-entry-hooks.sh

# Rest of the magic happens in the post-entry-hooks.sh script
