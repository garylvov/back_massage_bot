# Copyright (c) 2025, Gary Lvov, Vinay Balaji, Tim Bennet, Xandar Ingare, Ben Yoon
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from enum import Enum

import cv2
import numpy as np


class AreaColorsMapping(Enum):
    """
    Enum mapping body parts to their corresponding color ranges in HSV space.
    Each body part has a list of (lower_bound, upper_bound) tuples.
    """

    BACK = [((114, 42, 25), (135, 56, 14))]

    LEGS = [((142, 109, 7), (129, 121, 15))]


def get_close_color_mask(image, target_colors, threshold=30):
    """
    Finds a mask of all regions in the image that match any of the target BGR colors within a given threshold.

    :param image: Input image (BGR format, as read by OpenCV).
    :param target_colors: A tuple (B, G, R) for a single color or a list of such tuples for multiple colors.
    :param threshold: Intensity difference allowed (default: 30).
    :return: Binary mask where matching pixels are white (255) and others are black (0).
    """

    # Ensure target_colors is a list
    if not isinstance(target_colors, list):
        target_colors = [target_colors]

    # Convert image to numpy array
    image = np.array(image, dtype=np.int16)  # Prevents overflow when subtracting

    # Initialize an empty mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for target_color in target_colors:
        # Compute absolute difference from target color
        diff = np.abs(image - np.array(target_color, dtype=np.int16))

        # Check if all channels are within threshold
        color_mask = np.all(diff <= threshold, axis=-1).astype(np.uint8) * 255

        # Combine with existing mask using bitwise OR
        mask = cv2.bitwise_or(mask, color_mask)

    return mask
