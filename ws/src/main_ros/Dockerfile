FROM bmb_ubuntu22_python310:latest

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

SHELL ["conda", "run", "-n", "back_massage_bot", "/bin/bash", "-c"]
ARG USERNAME
USER ${USERNAME}
WORKDIR /ws/src/main_ros

# Install ROS with Conda
RUN conda clean -a -y && conda install -y mamba -c conda-forge && \
    conda config --env --add channels robostack-staging

# Install ROS, and a pinned version of realsense2-camera.
# Also, include some Kinova Dependencies so that mamba can solve the environment,
# While still respecting the pinned realsense version. Annoying but seemingly necessary.
# Also, don't change pytorch or cuda version.
RUN mamba install -y ros-humble-desktop ros-humble-realsense2-camera==4.54.1 \
    ros-humble-moveit \
    ros-humble-control-msgs \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-xacro

RUN mamba install -y compilers cmake \
                pkg-config make ninja colcon-common-extensions \
                catkin_tools rosdep

# Get deps for workspace
RUN rosdep init && rosdep update

# Copy ROS workspace
COPY ws/src/main_ros /ws/src/main_ros/

RUN rosdep install --from-paths /ws/src/main_ros --ignore-src -r -y --skip-keys="python3-typing-extensions"

# Build with explicit Python paths for the specific user
RUN source post-entry-hooks.sh
