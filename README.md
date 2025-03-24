# Combine a Robot Arm, a Sensor that creates Point Clouds, and a massage gun to give back massages.

This monorepo completes EECE 4792 - Electrical and Computer Engineering Capstone 2 at Northeastern for [Gary Lvov](https://garylvov.com), [Tim Bennett](https://www.linkedin.com/in/timothy-o-bennett/), [Xander Ingare](https://www.linkedin.com/in/alexander-ingare/), [Ben Yoon](https://www.linkedin.com/in/ben-yoon2003/), and [Vinay Balaji](https://www.linkedin.com/in/vinayajiteshbalaji/).

Our group extends our utmost gratitude to [Nathalie Hager](https://www.linkedin.com/in/nathalie-hager/) for gifting Gary the Kinova Jaco 2 arm that made this project possible. 

Although this project heavily targets the Kinova Jaco 2 robot arm and Intel RealSense L515 depth camera platforms, it could
be easily adapted to other arms as long as they support Cartesian planning with [MoveIt2](https://moveit.picknik.ai/main/index.html),
and other sensors, as long as they provide point cloud information. The base vision and motion planning code in ``back_massage_bot`` wouldn't need to change 
at all; only the ROS 2 integration in ``ws/src/main_ros`` and would have to be adapted to build and launch the new platform.

This repository provides a high level overview of the project, but you can also read [our final report here](garylvov.com)
if this README is not enough info. (TODO: Add report link).

# Installation

#### Pre-Requisites / System Information

This project is designed to be ran on an ```x86``` Linux computer within Docker containers running the NVIDIA container toolkit (CUDA/GPU Passthrough), so Linux/an NVIDA GPU is required.
We've tested on Ubuntu 22.04, but it should work on other Ubuntu versions as well.
If your Ubuntu/Linux computer has an NVIDIA GPU that's not yet configured, set it up along with the NVIDIA Docker container toolkit according to [these instructions](https://github.com/garylvov/dev_env/tree/main/setup_scripts/nvidia).


#### Cloning the Repo and Submodules
To get started, run the following.
```
git clone https://github.com/garylvov/back_massage_bot/ && \
cd back_massage_bot && git submodule init && git submodule update && \
sudo apt install pre-commit && pre-commit install && sudo apt install cppcheck cpplint clang-format # Optional for enforcing linting
```
#### Project Permissions (USB and Network)

To run this project, you may need to enable certain permissions. Make sure to unplug the camera and the robot arm prior to running
so that permissions are set properly.
For permissions, review ```set_permissions.sh``` before running it with the following.
```
sudo bash set-permissions.sh
```

### Docker Dev Guide (EZPZ)

We use Docker to simplify the installation of this repo. All dependencies are automatically installed in the Docker image from
this repository alone to prevent painstaking setup steps.

Using the stored image from Dockerhub for ROS 2 and Python/C++ development, run the following. Make sure that all
USB devices are plugged in prior to running the following. If a device is unplugged, then the container needs to be restarted.
```
bash pull_run_overlay.sh
```

You should now have a terminal from where to run commands, with all dependencies installed.
Your computer directories are symlinked into the container so local changes in the cloned repo are reflected within the container when running. 
The ``back_massage_bot`` directory is symlinked into the container at ``/back_massage_bot``,
and the ``ws/src`` directory is symlinked into the container at ``/ws/src``.
Once inside of the container, you should source the post entry hooks prior to running any commands as following.
```
source post-entry-hooks.sh
```

If you'd like to enter a new terminal window within an existing running container, you can run the following in a new window.
```
python3 docker.py --dive
```

To launch the entire system (either within the container, or outside), assuming that everything is plugged in (L515, Kinova, Massage ESP 32) prior
to starting the container, and the permissions were set earlier, run the following.

```
bash entrypoint.sh
```

If a dependency is desirable to prototype with, it can be temporarily added in this project's top level ``Dockerfile`` until it can be added to the stored image in Dockerhub (built from ```ws/src/main_ros/Dockerfile```)

<details>
<summary>Click to see how to build and upload images to Dockerhub </summary>
For ROS 2 and Python/C++ development with the image to be pushed to Dockerhub in the future, run the following.

```
bash build.sh && bash develop.sh # (for pure Python with no ROS, run bash back_massage_bot/develop.sh)
```

To push an updated image to Dockerhub, run the following (only works for @garylvov due to Dockerhub permissions, but provided for reference).

```
bash build_and_push_image.sh -h
```
</details>


# Model Training To Find Massage Candidate Regions

We transform the point cloud of the massage table and person into a 2D binary occupancy grid to create a low-dimensional environmental representation, making it easier to train instance segmentation models that identify massage candidate regions on the human body. 
This approach offers key advantages over direct point cloud segmentation: the low-dimensional nature of these representations enables straightforward synthetic generation of automatically labeled training data, as binary images are simpler to generate/auto label than point clouds. Segmentation bounding boxes are transformed to identify massage regions in 3D in the original point cloud.

![plot](assets/cloud_to_grid.gif)

Our model is a simple fine-tuned ``Yolov11`` from Ultralytics, trained exclusively on synthetic data, with good zero-shot transfer
performance on real data.

Fine tuned weights are available in the ``back_massage_bot/src/back_massage_bot/models`` directory. We trained on 200k synthetically
generated images on [@garylvov 's beloved 4 GPU rig](https://garylvov.com/projects/minerva/).

All training is done within the Docker container. Training can be done either in the base python only docker image
(``bash back_massage_bot/build.sh && bash back_massage_bot/develop.sh``) or in the overall project image (``bash pull_run_overlay.sh``).

To replicate our training process, generate the synthetic data, and then launch training.

The generate synthetic training images, run the following.

```
# Generate 100k samples training data on all cores
python3 /back_massage_bot/src/back_massage_bot/synthetic_data_gen.py --num_workers $(nproc) --dir /back_massage_bot/src/back_massage_bot/data/train --num_samples 100000 
# Generate validation dataset
python3 /back_massage_bot/src/back_massage_bot/synthetic_data_gen.py --num_workers $(nproc) --dir /back_massage_bot/src/back_massage_bot/data/valid --num_samples 1000
```

To train the network, run the following. You may wish to adjust the devices in the script
as well as the batch size to reflect your hardware configuration. The default configuration is for ``4x 24Gb`` GPUs.

```
python3 /back_massage_bot/src/back_massage_bot/train.py
```

We ended up doing some aggressive fine-tuning in a second stage of 100k images (for 200k images total) to get around imperfections in the synthetic data, starting with the previously trained model. We realized that the synthetic dataset overshoots the size of the torso, and doesn't detect the
head, so in the fine tuning model, we turn off circular noise that could be mistaken for the head, shrink the size of the torso bounding box, and make the ultralytics augmentations less aggressive. We continued from the previous model checkpoint to save training time; it is likely
that simply running the fine-tuning script from scratch (base pre-trained YOLOv11)would have yielded similar results.

```
# Get rid of the existing data
rm -rf /back_massage_bot/src/back_massage_bot/data/train/ && rm -rf /back_massage_bot/src/back_massage_bot/data/valid/
# Generate 30k samples fine-tune data on all cores
python3 /back_massage_bot/src/back_massage_bot/synthetic_data_gen.py --num_workers $(nproc) --dir /back_massage_bot/src/back_massage_bot/data/train --num_samples 100000 --fine-tune-torso
# Generate validation dataset
python3 /back_massage_bot/src/back_massage_bot/synthetic_data_gen.py --num_workers $(nproc) --dir /back_massage_bot/src/back_massage_bot/data/valid --num_samples 1000 --fine-tune-torso
# Start fine-tuning (more like retraining tbh ;) ) 
python3 /back_massage_bot/src/back_massage_bot/fine_tune.py
```

We tried to use zero-shot pretrained models, to no success, which motivated our custom yolo fine-tuning.

- We first tried Facebook's DensePose. 
 We found that the segmentation performance when the person wasn't facing the camera was poor,
and on the massage table, the person never faces the camera. 
  Additionally, we found that the L515 RGB isn't well aligned with the depth despite our best efforts, making it unsuitable 
  for our point cloud-based motion planning
- We tried Human3D (Takmaz, Schult, et. al ICCV 2023) segmentation. We found it difficult to run inference with the repository, and 
when we did get it to work, it failed to identify any humans in our point cloud, likely due to sampling differences between the 
synthetic data and our point cloud.
- We also tried PointSAM to use Segment Anything on Point Clouds; this couldn't identify unique regions.


# Motion Planning on Massage Candidate Regions

TODO: We use MoveIt. Our motion planning does rely on knowing two extrinsic transforms (see the next section for more information).



# Extrinsic Determination (Depth Camera to Robot Transform, and Massage Gun to Robot Transform)

The depth camera to robot extrinsic transform, needed to transform percieved points from the camera to points 
to visit with the robot arm, is found by a tool that @garylvov created. 
This tool is in a prototype stage, and unfortunately can't yet be released. 
This tool will be released by September 2025.
See [this page](https://garylvov.com/projects/extrinsic_cal/) for a teaser.

The tool works by constructing a synthetic robot mesh using the robot’s URDF and joint positions, 
then applies ICP to align the mesh with the depth sensor’s view of the robot to solve for the extrinsic transform.

If the tool is not present on the system (runs within its own docker container), the default config file, 
```back_massage_bot/src/back_massage_bot/config/default_transforms.yaml``` is used instead. 

To populate the default config file without the proprietary tool, set up the camera in a known location relative to the base of the arm, or measure the camera location, or perform some eye-to-hand calibration routine (such as placing an april tag near the base of the arm 
at a known location, then localizing from the camera to the april tag).

The massage gun to robot hand extrinsic transform was measured from CAD and confirmed in the real world. If the CAD from ``hardware/cad``
is used, the massage gun is the same/oriented the same as ours, then the values within the default config file ```back_massage_bot/src/back_massage_bot/config/default_transforms.yaml``` should be sufficient.

The default file also includes a niave crop of the massage table in the robot's base frame to enable easier downstream processing.


# Hardware Setup

We set up a camera on a tripod, as high as possible, facing downwards towards a person laying on a massage
table to get as close to a top-down view as possible. The robot arm is placed at a rough known location relative to the massage table.
TODO: Add more information here

We modified a massage gun to be powered by wire so that we don't need to rechage the battery.

We designed/3D printed two adapters found in ```hardware/CAD``` to mount a massage gun to the 
Kinova. We also zip-tied the mount a lot to ensure that there was no movement. We also
added drilled a pin to prevent rotation of the massage gun within the adapted.

TODO: Add more information about our hardware/ESP32 wiring here, and our table setup.


# Project Structure Overview

We intentionally seperate purely pythonic dependencies in ```back_massage_bot```, as to not require ROS 
for things such as model training. The ROS image is built on top of the python only image, also building the
```ws/src``` ROS workspace.

```
back_massage_bot/ # PROJECT_DIR
|-ws/src/ # Houses all ROS 2 packages. Each group of packages must have a corresponding dockerfile.
|------/main_ros/ # Example package parent folder, all subfolders share dependencies
|---------------/kinova-ros2/ # Kinova Submodule (May be graduated to forked copy)
|---------------/back_massage_bot_ros2/ # TODO: uses back_massage_bot python lib
|---------------/Dockerfile
|---------------/build.sh # builds ROS and the workspace on top of the python only dev
|---------------/develop.sh # Run ROS/C++ dev on top of python only dev
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
|-------------/build.sh # to build python only dev
|-------------/develop.sh # To run python only dev
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
|-pull_run_overlay.sh # Pull the pre-built docker hub image, and run it.
|-Dockerfile # temporary deps, until they can be graduated to the pre-built image.
|-build.sh # Build all docker containers for this project from source with no cache
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
The rules should be reflected in ``hardware``, and should be automatically installed by ``set-permissions.bash``.


# Troubleshooting

As a temporary measure, the firewall can be disabled with the following.
This only is needed if ```ros2 topic echo <TOPIC>``` doesn't work while the topic shows up
with ```ros2 topic list```, or the ROS 2 daemon keeps crashing.
```
sudo ufw disable # run sudo ufw enable when finished.
```
