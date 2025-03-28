#!/usr/bin/env python3
# Copyright (c) 2025, Gary Lvov, Tim Bennett, Xander Ingare, Ben Yoon, Vinay Balaji
# All rights reserved.
#
# SPDX-License-Identifier: MIT


from ultralytics import YOLO
import os
import glob
import shutil

# Create output directory for results
output_dir = "detection_results"
os.makedirs(output_dir, exist_ok=True)

# Load the model with the correct paths
model = YOLO("runs/detect/train12/weights/best.pt")  # trained model
# Set data configuration directly in args dict, not as an attribute
model.args['data'] = "data/data.yaml"

# Get all PNG files from a directory
input_dir = "grid_images"  # Change this to your input directory path
png_files = glob.glob(os.path.join(input_dir, "*.png"))

# Run batched inference on all PNG files
results = model(png_files)  # return a list of Results objects

# Process results list
for i, result in enumerate(results):
    # Get original filename without directory and extension
    orig_filename = os.path.basename(png_files[i])
    base_filename = os.path.splitext(orig_filename)[0]
    
    # Save the visualization with bounding boxes
    result_path = os.path.join(output_dir, f"{base_filename}_bbox.jpg")
    result.save(filename=result_path)
    
    # For debugging/viewing
    # result.show()
    
    # Optionally save the raw detection data
    boxes = result.boxes
    if boxes is not None:
        # Save box coordinates as text
        txt_path = os.path.join(output_dir, f"{base_filename}_boxes.txt")
        with open(txt_path, "w") as f:
            for box in boxes:
                # Write class, confidence, and coordinates (x1, y1, x2, y2)
                cls = int(box.cls.item())
                conf = box.conf.item()
                coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                f.write(f"Class: {cls}, Conf: {conf:.4f}, Coords: {coords}\n")

print(f"Saved all detection results to {output_dir}/")

model.export(format="onnx")