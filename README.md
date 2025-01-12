# therapeutech
Combine a Kinova Jaco 2, an Intel RealSense L515, and a massage gun to give back massages.

# Installation
This project is designed for Ubuntu Linux. If your computer has NVIDIA GPUs, set them up along with the NVIDIA container toolkit according to [these instructions](https://github.com/garylvov/dev_env/tree/main/setup_scripts/nvidia). If your computer doesn't have GPUs, [install docker](https://docs.docker.com/desktop/setup/install/linux/) and [docker compose](https://docs.docker.com/compose/install/linux/#install-using-the-repository).

To get started, run the following.
```
export TTECHDIR=$PWD/therapeutech &&
git clone https://github.com/garylvov/therapeutech/ &&
pip3 install pre-commit && pre-commit install # Optional for enforcing linting
```

To run this project, you may need to enable certain permissions, and you may need to disable your computer's firewall for UDP message transport between containers.
For permissions, review ```set_permissions.sh``` before running it with
```
sudo bash set_permissions.sh
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
|------/arm_and_cam/ # Example package parent folder, all subfolders share dependencies
|------------------/arm/ # Arm ROS 2 deps (potentially submodule)
|------------------/camera/ # Camera ROS 2 deps (potentially submodule)
|------------------/Dockerfile
|------------------/build.sh
|------------------/develop.sh
|------------------/post-entry-hooks.sh
|------------------/entrypoint.sh
|------------------/deploy.sh
|------/user_interfaces/
|-therapeutech/ # Where Python Stuff Lives
|-------------/external/ # Where third-party things that can't be easily pip installed live
|----------------------/NOT_PIP_INSTALLABLE_GIT_SUBMODULE_PLACEHOLDER.txt
|-------------/src/
|-----------------/__init__.py
|-------------/pyproject.toml
|-------------/requirements.txt
|-------------/Dockerfile
|-------------/build.sh
|-------------/develop.sh
|-------------/post-entry-hooks.sh
|-------------/entrypoint.sh
|-------------/deploy.sh
|-src/ # Where C++ Stuff Lives
|----/external/ # Where third-party things that can be easily apt installed live
|-------------/NOT_APT_INSTALLABLE_GIT_SUBMODULE_PLACEHOLDER.txt
|----/project_placeholder/
|------------------------/project_placeholder.cmake
|----/Dockerfile
|----/build.sh
|----/develop.sh
|----/post-entry-hooks.sh
|----/entrypoint.sh
|----/deploy.sh
|-hardware/
|---------/USB_README.md/ # How to deal with USB Devices described
|---------/10-camera.rules
|---------/50-esp.rules
|---------/60-arm.rules
|---------/set_usb_rules.sh # Set USB rules and permissions
|docker.py # Thin wrapper to simplify docker commands
|build.sh # Build all docker containers for this project
|develop.sh # Launch all docker containers for this project, without entrypoints in interactive mode
|post-entry-hooks.sh
|deploy.sh # Launch all docker containers for this project, and launch their entrypoints
|set-permissions.sh # Setup usb permissions
| README.md # You are here
```

# Contribution Guidelines
#### Overall Workflow
To merge code into main, first open a branch from main.
This branch will be where your changes are housed.
Feel free to make commits to your branch at your leisure
When your code is ready to be merged into main, open a pull request from your branch into main.
Prior to opening a PR, check that everything pases the style guide with ```pre-commit run --all-files```

### Docker Guide
If there is a directory you'd like to develop in (PARENT_DEV), run the following.

```
export TTECHDIR=<PROJECT_DIRECTORY> && bash <PARENT_DEV>/build.sh && bash <PARENT_DEV>/develop.sh
```

Once inside of the container, run
```
bash post-entry-hooks.sh
```

#### Code Guidelines
- For each new functionality, please make sure to update the README.
- For each piece of software that includes any third-party dependencies, please include a ``Dockerfile`` in the parent development (<PARENT_DEV> directory).
  The ``Dockerfile`` should auto-install any third-party dependencies. The work directory of the Dockerfile should include a copied over ``post-entry-hooks.sh`` and ``entrypoint.sh``
  -  Please include a ``build.sh`` that has the command to build the ``Dockerfile`` from the topmost folder (PROJECT_DIR), like (``docker build -t PARENT_DEV --file<PROJECT_DEV_TO_PARENT_DEV_PATH>/Dockerfile``).
  -  Please include a ``develop.sh`` that has the command to enter your docker container in interactive mode, with the directories symlinked to the main computer so that
     changes made outside of the docker container are also reflected within the docker container.
     See ```python3 docker.py``` for a template on how to achieve this.
     For example, this could look like the following.
      ```
      python3 $TTECHDIR/docker.py PARENT_DEV -i -v <PATH_TO_PARENT_DEV_FROM_PROJECT_DIR>
      ```
  - Please include a ``post-entry-hooks.sh``. If something isn't possible to set persistently in an ```ENV``` variable in the Dockerfile while in interactive mode, put it here.
    For example, this could include ```source /opt/ros/humble/setup.bash```.
  - Please include a ``entrypoint.sh``. For example, this may lool like the following.
    ```
    bash post-entry-hooks.sh && <SOME_ROS_COMMAND_HERE> & # Detach for next command
    bash post-entry-hooks.sh && <SOME_ROS_COMMAND_HERE> & # Detach for next command
    ```
  - Please include a ``deploy.sh`` that has the command to run your container, as well as the related code.
    ```
    python3 $TTECHDIR/docker.py PARENT_DEV -e
    ```

#### Integration with Hardware Devices
All USB devices should have [static rules](https://msadowski.github.io/linux-static-port/).
The rules should be reflected in ``/hardware``, and should be automatically installed by ``set_permissions.sh``.
