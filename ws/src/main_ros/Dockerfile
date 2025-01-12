FROM tt_ubuntu22_python310:latest

ARG USERNAME

USER ${USERNAME}

WORKDIR /ws/src/main_ros
COPY ws/src/main_ros /ws/src/main_ros/

# Patch for Kinova driver
USER root
RUN for lib in USB{Comm,Command}LayerUbuntu.so Eth{Comm,Command}LayerUbuntu.so; do ln -fs x86_64-linux-gnu/$lib /ws/src/main_ros/kinova-ros2/kinova_driver/lib/$lib; done
USER ${USERNAME}


RUN source /ws/src/main_ros/post-entry-hooks.sh && conda install -y mamba -c conda-forge && \
conda config --env --add channels robostack-staging && \
mamba install -y python=3.11 ros-humble-desktop && \
mamba deactivate && \
mamba activate therapeutech && \
mamba install -y compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep && \
rosdep install --from-paths /ws/src/main_ros --ignore-src -r -y 

RUN cd /ws/ && colcon build --symlink-install 