## Combine a Kinova Jaco 2, an Intel RealSense L515, and a massage gun to give back massages.

(Click on the arrow pointing to section titles to expand ``README.md``)

<details><summary><b>Pre-Requisites / System Information </b></summary>


This project is designed to be ran within Ubuntu Docker containers with CUDA support.
Ideally, this means the project is running from within a container on an Ubuntu computer with an NVIDIA GPU.
There is a workaround for limited development on other platforms with Conda.
However, Ubuntu and/or Docker is strongly recommended, as certain libraries may not be compatible with other systems,
resulting in incompatibility/reduced features sets.

- If your Ubuntu/Linux computer has an NVIDIA GPU, set it up along with the NVIDIA Docker container toolkit according to [these instructions](https://github.com/garylvov/dev_env/tree/main/setup_scripts/nvidia).

- If your Ubuntu/Linux computer doesn't have GPUs, just [install docker](https://docs.docker.com/desktop/setup/install/linux/).
- If you have a Mac [install Docker](https://docs.docker.com/desktop/setup/install/mac-install/). [Also, try to set up display forwarding](https://gist.github.com/sorny/969fe55d85c9b0035b0109a31cbcb088). You may need to modify the display flags in ``docker.py`` to match the display forwarding tutorial.

- If your computer has an NVIDIA GPU but runs windows, you may be able to use [these instructions](https://forums.developer.nvidia.com/t/guide-to-run-cuda-wsl-docker-with-latest-versions-21382-windows-build-470-14-nvidia/178365/10.) to set up Docker. [Also, try to set up display forwarding](https://stackoverflow.com/questions/61110603/how-to-set-up-working-x11-forwarding-on-wsl2). That being said, you may have an easier time just dual-booting Ubuntu.

### If Docker isn't working, or display forwarding isn't working, you can use the reduced feature set with Conda. Install [miniforge](https://conda-forge.org/download/) and follow the No Docker section of this ``README``.

</details>

<details>

<summary><b>
Installation
</b></summary>

To get started, run the following.
```
git clone https://github.com/garylvov/back_massage_bot/ &&
cd back_massage_bot && git submodule init && git submodule update && \
pip3 install pre-commit && pre-commit install # Optional for enforcing linting
```
</details>

<details>
 <summary><b>Ubuntu Only Permissions (Alternative platforms will have different steps)</b></summary>

To run this project, you may need to enable certain permissions, and you may need to disable your computer's firewall for UDP message transport between containers.
For permissions, review ```set_permissions.sh``` before running it with the following.
```
sudo bash set-permissions.sh
```
As a temporary measure, the firewall can be disabled with the following.
```
sudo ufw disable
```
Always re-enable the firewall after finishing deployment/development with the following.
```
sudo ufw enable
```
</details>

<details> <summary><b> Project Structure Overview </b></summary>

```
back_massage_bot/ # PROJECT_DIR
|-ws/src/ # Houses all ROS 2 packages. Each group of packages must have a corresponding dockerfile.
|------/main_ros/ # Example package parent folder, all subfolders share dependencies
|---------------/kinova-ros2/ # Kinova Submodule (may be graduated to forked copy)
|---------------/realsense-ros/ # Realsense Submodule
|---------------/back_massage_bot_ros2/ # TODO: uses back_massage_bot python lib
|---------------/Dockerfile
|---------------/build.sh
|---------------/develop.sh
|---------------/post-entry-hooks.sh
|---------------/entrypoint.sh
|------/user_interfaces/
|-back_massage_bot/ # Where Python Stuff Lives
|-------------/external/ # Where third-party things that can't be easily pip installed live
|----------------------/NOT_PIP_INSTALLABLE_GIT_SUBMODULE_PLACEHOLDER.txt
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
</details>


<details> <summary><b> Docker Guide (EZPZ, Strongly Recommended) </b></summary>

For pure python development (no ROS), run the following:
```
bash back_massage_bot/build.sh && bash back_massage_bot/develop.sh
```

For ROS 2 and Python development (we can add C++ if needed), run the following.
```
bash ws/src/main_ros/build.sh && bash ws/src/main_ros/develop.sh
```

You should now have a terminal from where to run commands, with all dependencies installed.
Your computer directories are symlinked into the container so local changes in the cloned repo are reflected within the container when running.

Once inside of the container, you may need to run the following prior to commands.
```
bash post-entry-hooks.sh
```

If you'd like to enter a new terminal window within an existing container, you can run the following in a new window.
```
python3 docker.py --dive
```
</details>

<details> <summary><b> No Docker Guide ( Limited Functionality, no automatic install ) </b></summary>

The docker method is strongly recommended.
However, you may be able to run parts of this project locally due to [RoboStack](https://robostack.github.io/GettingStarted.html) being largely cross platform for
the sake of local development on Mac or Windows.

#### To run locally, first install [miniforge](https://conda-forge.org/download/)

Then, run the following to create the conda environment.
```
conda env create -f back_massage_bot/environment.yml && \
conda activate back_massage_bot && \
pip install -e back_massage_bot && \
```

Install MMPose.

```
# within conda
bash back_massage_bot/post-entry-hooks.sh
```

If this doesn't work, follow the installation in [MMPose](https://mmpose.readthedocs.io/en/latest/installation.html)
to install within the existing conda environment. You may need to use ``mmcv-lite``.
You may need to switch to CPU only pytorch with
```
# First remove existing pytorch and torchvision
conda remove pytorch torchvision

# Then install CPU-only versions
conda install pytorch torchvision cpuonly -c pytorch
```

Then install ROS.
```
conda clean -a -y && \
conda install -y mamba -c conda-forge && \
conda config --env --add channels robostack-staging && \
mamba install -y python=3.11 ros-humble-desktop && \
conda deactivate && \
conda activate back_massage_bot && \
mamba install -y compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep
```

Then, install the ROS workspace dependencies.
```
rosdep init && rosdep update && \
rosdep install --from-paths ws/src/main_ros --ignore-src -r -y
```

Then, build the ROS workspace (with a ```kinova-driver``` patch)
```
for lib in USB{Comm,Command}LayerUbuntu.so Eth{Comm,Command}LayerUbuntu.so; do ln -fs x86_64-linux-gnu/$lib ws/src/main_ros/kinova-ros2/kinova_driver/lib/$lib; done && \
cd ws/ && \
colcon build --symlink-install
```

If the above doesn't work, you can can try to build without the kinova driver. You may need to remove other offending packages.
```
rm ws/src/main_ros/kinova-ros2/kinova_driver/ &&
cd ws/ && rosdep install --from-paths src/main_ros --ignore-src -r -y && \
colcon build --symlink-install &&
source ws/install/setup.bash
```

If the above doesn't work and complains about realsense versions, you can try the following workaround, then rebuild
```
# From within ws/ like above command changes to
sed -i 's/find_package(realsense2 2.55.1)/find_package(realsense2 2.54.1)/' src/main_ros/realsense-ros/realsense2_camera/CMakeLists.txt && \
colcon build --symlink-install &&
source ws/install/setup.bash
```
</details>

You now have a terminal. Make sure to run ``conda activate back_massage_bot && source ws/install/setup.bash``
prior to commands.
<details> <summary><b> Contribution Guidelines </b></summary>

- To merge code into main, first open a branch from main.
This branch will be where your changes are housed.
Feel free to make commits to your branch at your leisure.
When your code is ready to be merged into main, open a pull request from your branch into main.
Prior to opening a PR, check that everything pases the style guide with ```pre-commit run --all-files```.
- Development must be done within a Docker container and/or a Conda environment.
Docker containers running on Ubuntu Linux with CUDA are considered the highest source of truth.
- **Do not source system ROS within the docker container or conda environment** (this is because of our use of [RoboStack](https://robostack.github.io/GettingStarted.html).)
- For each new functionality, please make sure to update the ``README.md``.
- For each new piece of software that may include third-party dependencies, please include a reproducible method to automatically build/install that software along with its dependencies. This can be achieved through modifying an existing ``Dockerfile`` or Conda ```environment.yml```, or maybe adding a new GitHub Submodule. See ``ws/src/main_ros`` for an example of automatic building/installation of software with dependencies.
If the software conflicts with existing Docker images, please provide a new directory with a new ``Dockerfile`` for that software. As long as ROS 2 is installed in all Docker images, several images can be used cohesively while communicating with each other.
  The ``Dockerfile`` should auto-install any third-party dependencies along with the software. The work directory of the Dockerfile should include a copied over ``post-entry-hooks.sh`` and ``entrypoint.sh``
  -  Please include a ``build.sh`` that has the command to build the ``Dockerfile`` from the topmost folder (like in
      ``ws/src/main_ros/build.sh``).
  -  Please include a ``develop.sh`` that has the command to enter your docker container in interactive mode, with the directories symlinked to the main computer so that
     changes made outside of the docker container are also reflected within the docker container (like in
      ``ws/src/main_ros/develop.sh``)
  - Please include a ``post-entry-hooks.sh``. If something isn't possible to set persistently in an ```ENV``` variable in the Dockerfile while in interactive mode, put it here, like in ``ws/src/main_ros/post-entry-hooks.sh``
  - Please include a ``entrypoint.sh`` that deploys all of the software once in the container, like in ``ws/src/main_ros/entrypoint.sh``
</details>

<details> <summary><b> Integration with Hardware Devices </b></summary>

All USB devices should have [static rules](https://msadowski.github.io/linux-static-port/).
The rules should be reflected in ``/hardware``, and should be automatically installed by ``set_permissions.sh``.
</details>
