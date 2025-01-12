# therapeutech
Combine a Kinova Jaco 2, an Intel RealSense L515, and a massage gun to give back massages.

# Installation
This project is designed for Ubuntu Linux, for within Docker containers.
There is a workaround for some development on platforms other than Ubuntu without Docker with Conda and [RoboStack](https://robostack.github.io/index.html).
However, Ubuntu and/or Docker is strongly recommended, as certain libraries (looking at you ``kinova_driver``) seem to rely on ``.so`` files pre-compiled for ``x86``.

If your computer has an NVIDIA GPU, set them up along with the NVIDIA container toolkit according to [these instructions](https://github.com/garylvov/dev_env/tree/main/setup_scripts/nvidia). If your computer doesn't have GPUs, [install docker](https://docs.docker.com/desktop/setup/install/linux/) and [docker compose](https://docs.docker.com/compose/install/linux/#install-using-the-repository) (unless you are using conda)

To get started, run the following.
```
git clone https://github.com/garylvov/therapeutech/ &&
cd therapeutech && git submodule init && git submodule update && \
pip3 install pre-commit && pre-commit install # Optional for enforcing linting
```

### Ubuntu Only (Alternative platforms will have different steps)
To run this project, you may need to enable certain permissions, and you may need to disable your computer's firewall for UDP message transport between containers.
For permissions, review ```set_permissions.sh``` before running it with
```
sudo bash set-permissions.sh
```
As a temporary measure, the firewall can be disabled with
```
sudo ufw disable
```
Always re-enable the firewall after finishing deployment/development with
```
sudo ufw enable
```
# Project Structure Overview
```
therapeutech/ # PROJECT_DIR
|-ws/src/ # Houses all ROS 2 packages. Each group of packages must have a corresponding dockerfile.
|------/main_ros/ # Example package parent folder, all subfolders share dependencies
|---------------/kinova-ros2/ # Kinova Submodule (may be graduated to forked copy)
|---------------/realsense-ros/ # Realsense Submodule
|---------------/therapeutech-ros2/ # TODO: uses therapeutech python lib
|---------------/Dockerfile
|---------------/build.sh
|---------------/develop.sh
|---------------/post-entry-hooks.sh
|---------------/entrypoint.sh
|------/user_interfaces/
|-therapeutech/ # Where Python Stuff Lives
|-------------/external/ # Where third-party things that can't be easily pip installed live
|----------------------/NOT_PIP_INSTALLABLE_GIT_SUBMODULE_PLACEHOLDER.txt
|-------------/src/therapeutech/ # Python Lib No ROS
|------------------------------/__init__.py
|-------------/pyproject.toml
|-------------/requirements.txt
|-------------/environment.yaml # For Conda
|-------------/Dockerfile
|-------------/build.sh
|-------------/develop.sh
|-------------/post-entry-hooks.sh
|-------------/entrypoint.sh
|-src/ # Where C++ Stuff Lives
|----/external/ # Where third-party things that can be easily apt installed live
|-------------/NOT_APT_INSTALLABLE_GIT_SUBMODULE_PLACEHOLDER.txt
|----/project_placeholder/
|------------------------/CMakeLists.txt
|----/Dockerfile
|----/build.sh
|----/develop.sh
|----/post-entry-hooks.sh
|----/entrypoint.sh
|-hardware/
|----------/CAD/ # All CAD related things for this project.
|---------/USB_README.md/ # How to deal with USB Devices described
|---------/10-camera.rules
|---------/50-esp.rules
|---------/60-arm.rules
|---------/set_usb_rules.sh # Set USB rules and permissions
|-docker.py # Thin wrapper to simplify docker commands
|-build.sh # Build all docker containers for this project
|-entrypoint.sh # Launch all docker containers for this project, and launch their entrypoints
|-set-permissions.sh # Setup usb permissions
|-README.md # You are here
```

# Workflows
### Overall Workflow
To merge code into main, first open a branch from main.
This branch will be where your changes are housed.
Feel free to make commits to your branch at your leisure.
When your code is ready to be merged into main, open a pull request from your branch into main.
Prior to opening a PR, check that everything pases the style guide with ```pre-commit run --all-files```.

### Docker Guide (Recommended)
If there is a directory you'd like to develop in (PARENT_DEV), run the following. This will ensure that changes that are made in the parent folder
on the host machine will be reflected with the container.

```
bash <PARENT_DEV>/build.sh && bash <PARENT_DEV>/develop.sh

# For example, for pure python development;
bash therapeutech/build.sh && bash therapeutech/develop.sh

# For example, for ROS 2 and Python development (we can add C++ if needed)
bash ws/src/main_ros/build.sh && bash ws/src/main_ros/develop.sh
```

Once inside of the container, run
```
bash post-entry-hooks.sh
```

If you'd like to enter a new terminal window within an existing container, you can run
```
python3 docker.py --dive
```

### No Docker Guide
The docker method is strongly recommended.
However, you may be able to run parts of this project locally due to [RoboStack](https://robostack.github.io/GettingStarted.html) being largely cross platform for
the sake of local development on Mac or Windows.

#### To run locally, first install [miniforge](https://github.com/conda-forge/miniforge)
Then, run the following
```
conda env create -f therapeutech/environment.yml && \
conda activate therapeutech && \
pip install -e therapeutech && \
conda clean -a -y && \
conda install -y mamba -c conda-forge && \
conda config --env --add channels robostack-staging && \
mamba install -y python=3.11 ros-humble-desktop && \
conda deactivate && \
conda activate therapeutech && \
mamba install -y compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep && \
for lib in USB{Comm,Command}LayerUbuntu.so Eth{Comm,Command}LayerUbuntu.so; do ln -fs x86_64-linux-gnu/$lib ws/src/main_ros/kinova-ros2/kinova_driver/lib/$lib; done && \
rosdep init && rosdep update && rosdep install --from-paths ws/src/main_ros --ignore-src -r -y && \
cd ws/ && colcon build --symlink-install
```

If the above doesn't work, you can get rid of the ``ws/src/main_ros/kinova-ros2/kinova_driver/`` directory and rerun
```
rosdep install --from-paths ws/src/main_ros --ignore-src -r -y && \
cd ws/ && colcon build --symlink-install
```

**Do not source system ROS within the docker container or conda environment** (this is because of our use of [RoboStack](https://robostack.github.io/GettingStarted.html).)

### Code Guidelines
- For each new functionality, please make sure to update the README.
- For each piece of software that includes any third-party dependencies, please include a ``Dockerfile`` in the parent development (<PARENT_DEV> directory).
  The ``Dockerfile`` should auto-install any third-party dependencies. The work directory of the Dockerfile should include a copied over ``post-entry-hooks.sh`` and ``entrypoint.sh``
  -  Please include a ``build.sh`` that has the command to build the ``Dockerfile`` from the topmost folder (PROJECT_DIR), like (``docker build -t PARENT_DEV --file<PROJECT_DEV_TO_PARENT_DEV_PATH>/Dockerfile .``).
  -  Please include a ``develop.sh`` that has the command to enter your docker container in interactive mode, with the directories symlinked to the main computer so that
     changes made outside of the docker container are also reflected within the docker container.
     For example, this could look like the following.
      ```
      python3 docker.py <NAME_FROM_BUILD.SH> -i -v <PATH_TO_PARENT_DEV_FROM_PROJECT_DIR>
      ```
  - Please include a ``post-entry-hooks.sh``. If something isn't possible to set persistently in an ```ENV``` variable in the Dockerfile while in interactive mode, put it here.
    For example, this could include ```source /opt/ros/humble/setup.bash```.
  - Please include a ``entrypoint.sh``. For example, this may lool like the following.
    ```
    bash post-entry-hooks.sh && <SOME_ROS_COMMAND_HERE> & # Detach for next command
    bash post-entry-hooks.sh && <SOME_ROS_COMMAND_HERE> & # Detach for next command
    ```

#### Integration with Hardware Devices
All USB devices should have [static rules](https://msadowski.github.io/linux-static-port/).
The rules should be reflected in ``/hardware``, and should be automatically installed by ``set_permissions.sh``.
