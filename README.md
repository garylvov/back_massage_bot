## Combine a Kinova Jaco 2, an Intel RealSense L515, and a massage gun to give back massages.

## Installation

#### Pre-Requisites / System Information

This project is designed to be ran on Ubuntu within Docker containers running the NVIDIA container toolkit (CUDA/GPU Passthrough).
We've tested on Ubuntu 22.04, but it should work on other Ubuntu versions as well.
If your Ubuntu/Linux computer has an NVIDIA GPU, set it up along with the NVIDIA Docker container toolkit according to [these instructions](https://github.com/garylvov/dev_env/tree/main/setup_scripts/nvidia).


#### Cloning the Repo and Submodules
To get started, run the following.
```
git clone https://github.com/garylvov/back_massage_bot/ &&
cd back_massage_bot && git submodule init && git submodule update && \
sudo apt install pre-commit && pre-commit install && sudo apt install cppcheck cpplint clang-format # Optional for enforcing linting
```
#### Project Permissions (USB and Network)

To run this project, you may need to enable certain permissions, and you may need to disable your computer's firewall for UDP message transport between containers.
For permissions, review ```set_permissions.sh``` before running it with the following.
```
sudo bash set-permissions.sh
```
As a temporary measure, the firewall can be disabled with the following.
This only is needed if ```ros2 topic echo <TOPIC>``` doesn't work while the topic shows up
with ```ros2 topic list```, or the ROS 2 daemon keeps crashing.
```
sudo ufw disable # run sudo ufw enable when finished.
```

### Docker Guide (EZPZ)

Using the stored image from Dockerhub for ROS 2 and Python/C++ development, run the following
```
bash pull_run_overlay.sh
```

For ROS 2 and Python development, run the following.
```
bash ws/src/main_ros/build.sh && bash ws/src/main_ros/develop.sh # (for pure Python with no ROS, run bash back_massage_bot/develop.sh)
```

You should now have a terminal from where to run commands, with all dependencies installed.
Your computer directories are symlinked into the container so local changes in the cloned repo are reflected within the container when running. Once inside of the container, you may need to run the following prior to commands (definitely prior to ``ros2 run``
or ``ros2 launch`` after making changes to ROS packages.) (Due to our use of [RoboStack](https://robostack.github.io/GettingStarted.html), do not source system ROS, only use the ```post-entry-hooks.sh```)
```
source post-entry-hooks.sh
```

If you'd like to enter a new terminal window within an existing container, you can run the following in a new window.
```
python3 docker.py --dive
```

# Project Structure Overview

```
back_massage_bot/ # PROJECT_DIR
|-ws/src/ # Houses all ROS 2 packages. Each group of packages must have a corresponding dockerfile.
|------/main_ros/ # Example package parent folder, all subfolders share dependencies
|---------------/kinova-ros2/ # Kinova Submodule (May be graduated to forked copy)
|---------------/back_massage_bot_ros2/ # TODO: uses back_massage_bot python lib
|---------------/Dockerfile
|---------------/build.sh
|---------------/develop.sh
|---------------/post-entry-hooks.sh
|---------------/entrypoint.sh
|------/user_interfaces/
|-back_massage_bot/ # Where Python Stuff Lives
|-------------/src/back_massage_bot/ # Python Lib No ROS
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
|----/Dockerfile
|----/build.sh
|----/develop.sh
|----/post-entry-hooks.sh
|----/entrypoint.sh
|-hardware/
|----------/CAD/ # All CAD related things for this project.
|---------/USB_README.md/ # How to deal with USB Devices described
|---------/99-realsense-libusb.rules
|---------/setup-udev-rules.bash # Set USB rules and permissions
|-docker.py # Thin wrapper to simplify docker commands
|-build.sh # Build all docker containers for this project
|-entrypoint.sh # Launch all docker containers for this project, and launch their entrypoints
|-set-permissions.sh # Setup usb permissions
|-README.md # You are here
```

# Contribution Guidelines

- To merge code into main, open a Pull Request.
Make sure to check formatting with ```pre-commit run --all-files```.
- Development should be done within a Docker container.
Docker containers running on Ubuntu Linux with CUDA are considered the highest source of truth.
- **Do not source system ROS within the docker container or conda environment** (this is because of our use of [RoboStack](https://robostack.github.io/GettingStarted.html).)
- For each new functionality, please make sure to update the ``README.md``.
- For each new piece of software that may include third-party dependencies, please include a reproducible method to automatically build/install that software along with its dependencies.
This can be achieved through modifying an existing ``Dockerfile``/```environment.yml```, or maybe adding a new GitHub Submodule.
See ``ws/src/main_ros`` for an example of automatic building/installation of software with dependencies.
If the software conflicts with existing Docker images, please provide a new directory with a new ``Dockerfile`` for that software.
As long as ROS 2 is installed in all Docker images, several images can be used cohesively while communicating with each other.
  The ``Dockerfile`` should auto-install any third-party dependencies along with the software.
  The work directory of the Dockerfile should include a copied over ``post-entry-hooks.sh`` and ``entrypoint.sh``
  -  Please include a ``build.sh`` that has the command to build the ``Dockerfile`` from the topmost folder (like in
      ``ws/src/main_ros/build.sh``).
  -  Please include a ``develop.sh`` that has the command to enter your docker container in interactive mode, with the directories symlinked to the main computer so that
     changes made outside of the docker container are also reflected within the docker container (like in
      ``ws/src/main_ros/develop.sh``)
  - Please include a ``post-entry-hooks.sh``.
  If something isn't possible to set persistently in an ```ENV``` variable in the Dockerfile while in interactive mode, put it here, like in ``ws/src/main_ros/post-entry-hooks.sh``
  - Please include a ``entrypoint.sh`` that deploys all of the software once in the container, like in ``ws/src/main_ros/entrypoint.sh``
- All USB devices should have [static rules](https://msadowski.github.io/linux-static-port/).
The rules should be reflected in ``/hardware``, and should be automatically installed by ``set-permissions.bash``.
