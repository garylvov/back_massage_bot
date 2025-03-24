#!/usr/bin/env python3
# Copyright (c) 2025, Gary Lvov, Tim Bennett, Xander Ingare, Ben Yoon, Vinay Balaji
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from ultralytics import YOLO
from pathlib import Path
model = YOLO('runs/detect/train12/weights/best.pt') #YOLO("/back_massage_bot/src/back_massage_bot/models/first_run/best.pt")
data_yaml = str(Path(__file__).parent / "data/data.yaml")

# Set some aggressive param
model.train(data=data_yaml,
           epochs=100,
           imgsz=224,
           batch=800, # Optimized for 96GB VRAM
           cache="disk",
           # Disable all augmentations for grayscale occupancy grid images
           # Color/intensity transformations
           hsv_h=0.0,        # Hue (range: 0.0-1.0, default: 0.015), 1.0 would allow hue shift by 360 degrees
           hsv_s=0.0,        # Saturation (range: 0.0-1.0, default: 0.7)
           hsv_v=0.0,        # Value/brightness (range: 0.0-1.0, default: 0.4)
           
           # Geometric transformations
           degrees=180.0,      # Rotation (range: 0.0-180.0, default: 0.0)
           translate=0.6,    # Translation (range: 0.0-1.0, default: 0.1)
           scale=0.3,        # Scaling (range: 0.0-1.0, default: 0.5)
           shear=0,        # Shear (range: 0.0-10.0, default: 0.0)
           perspective=0.001,  # Perspective distortion (range: 0.0-0.001, default: 0.0)
           
           # Flip transformations
           flipud=0.0,       # Vertical flip (range: 0.0-1.0, default: 0.0, probability)
           fliplr=0.0,       # Horizontal flip (range: 0.0-1.0, default: 0.5, probability)
           
           # Complex augmentations
           mosaic=0.0,       # Mosaic augmentation (range: 0.0-1.0, default: 1.0, probability)
           mixup=0.0,        # Mixup augmentation (range: 0.0-1.0, default: 0.0, probability)
           copy_paste=0.1,   # Copy-paste (range: 0.0-1.0, default: 0.0, probability)
           
           # Keep other parameters the same
           freeze=0,
           patience=10,
           plots=True,
           save=True,
           workers=8,
           device="0,1,2,3",)
