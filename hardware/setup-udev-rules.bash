#!/bin/bash
# https://github.com/IntelRealSense/librealsense/blob/master/scripts/setup_udev_rules.sh
echo "Copying realsense udev rules..." && \
sudo cp hardware/99-realsense-libusb.rules /etc/udev/rules.d/ && \
echo "Copying kinova udev rules..." && \
sudo cp ws/src/main_ros/kinova-ros2/kinova_driver/udev/10-kinova-arm.rules /etc/udev/rules.d/ && \
echo "Copying ESP32 udev rules..." && \
sudo cp hardware/99-esp32-s3-box.rules /etc/udev/rules.d/ && \
echo "Copying Arduino udev rules..." && \
sudo cp hardware/99-arduino-massager.rules /etc/udev/rules.d/ && \
echo "Applying changes..." && \
sudo /lib/systemd/systemd-udevd --daemon && \
sudo udevadm control --reload-rules && sudo udevadm trigger
