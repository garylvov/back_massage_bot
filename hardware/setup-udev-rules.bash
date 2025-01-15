#!/bin/bash
# https://github.com/IntelRealSense/librealsense/blob/master/scripts/setup_udev_rules.sh
sudo cp hardware/99-realsense-libusb.rules /etc/udev/rules.d/ && \
sudo /lib/systemd/systemd-udevd --daemon && \
sudo udevadm control --reload-rules && sudo udevadm trigger
