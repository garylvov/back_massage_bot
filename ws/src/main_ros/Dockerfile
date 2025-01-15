FROM bmb_ubuntu22_python310:latest
SHELL ["conda", "run", "-n", "back_massage_bot", "/bin/bash", "-c"]
RUN which python

ARG USERNAME
USER ${USERNAME}
WORKDIR /ws/src/main_ros

# Install ROS with Conda
RUN conda clean -a -y && conda install -y mamba -c conda-forge && \
conda config --env --add channels robostack-staging
RUN mamba install -y python ros-humble-desktop
RUN mamba install -y compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep

# Install Realsense
RUN mamba install -y ros-humble-realsense2-camera==4.54.1
# Get deps for workspace
RUN rosdep init && rosdep update

# Copy ROS workspace
COPY ws/src/main_ros /ws/src/main_ros/

# Install dependencies
RUN rosdep install --from-paths /ws/src/main_ros --ignore-src -r -y

# Patch for Kinova driver
USER root
RUN for lib in USB{Comm,Command}LayerUbuntu.so Eth{Comm,Command}LayerUbuntu.so; do \
    ln -fs x86_64-linux-gnu/$lib /ws/src/main_ros/kinova-ros2/kinova_driver/lib/$lib; \
    done

# Switch back to user and build workspace
USER ${USERNAME}
RUN cd /ws/ && colcon build --symlink-install
