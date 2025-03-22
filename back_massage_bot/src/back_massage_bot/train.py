#!/usr/bin/env python3
# Copyright (c) 2025, Gary Lvov, Tim Bennett, Xander Ingare, Ben Yoon, Vinay Balaji
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolo11m.pt") 
data_yaml = str(Path(__file__).parent / "dataset/data.yaml")  

model.train(data=data_yaml, 
            epochs=100, 
            imgsz=800, 
            batch=100,
            cache="disk",
            freeze=0, 
            copy_paste=.8,
            hsv_v=.3,
            erasing=.9,
            crop_fraction=.8,
            translate=.9,
            mixup=.4,
            perspective=0.00005,
            patience=20, 
            plots = True, 
            save=True, 
	        workers = 8, 
	        device="0,1,2,3",)