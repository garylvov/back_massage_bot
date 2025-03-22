#!/usr/bin/env python3
# Copyright (c) 2025, Gary Lvov, Tim Bennett, Xander Ingare, Ben Yoon, Vinay Balaji
# All rights reserved.
#
# SPDX-License-Identifier: MIT


import numpy as np

"""
Through transforming a point cloud into a 2D binary occupancy grid, we create a low-dimensional representation
of the environment that can be leveraged to easily train an instance segmentation model to 
identify different massage candidate regions.

Point clouds are transformed into images as opposed to learning a point cloud segmentation model for the following reasons:
 - It's harder to label point clouds than images
 - Since the images are so low-dimensional, it's easy to learn the model, and it's easy to synthetically generate images
    with associated labels automatically.
"""


def create_binary_occupancy_grid_from_planar_point_cloud(points: np.ndarray,
                                            crop_bounds: dict[str, float],
                                            grid_resolution: float) -> tuple:
        """Create a 2D binary occupancy grid from a point cloud.
        The occupancy grid is created along the XY plane with naive cropping bounds, and the Z coordinate is ignored.
        For other desired occupancy grid orientations, project the point cloud to a desired plane prior to calling this function."""
        try:
            # Calculate grid dimensions based on crop bounds
            x_min, x_max = crop_bounds["x_min"], crop_bounds["x_max"]
            y_min, y_max = crop_bounds["y_min"], crop_bounds["y_max"]

            # Calculate grid size
            grid_width = int(np.ceil((x_max - x_min) / grid_resolution))
            grid_height = int(np.ceil((y_max - y_min) / grid_resolution))

            # Create empty grid
            grid = np.zeros((grid_width, grid_height), dtype=bool)

            # Fill grid cells
            for point in points:
                # Calculate grid cell indices
                x_idx = int((point[0] - x_min) / self.grid_resolution)
                y_idx = int((point[1] - y_min) / self.grid_resolution)

                # Check if the indices are within grid bounds
                if 0 <= x_idx < grid_width and 0 <= y_idx < grid_height:
                    grid[x_idx, y_idx] = True

            return grid, (x_min, y_min)
        
        except Exception as e:
            raise ValueError(f"Error creating occupancy grid: {str(e)}")