#!/usr/bin/env python3

# Copyright (c) 2025, Gary Lvov, Vinay Balaji, Tim Bennet, Xandar Ingare, Ben Yoon
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import cv2
import numpy as np
import torch
from densepose import add_densepose_config
from densepose.vis.base import CompoundVisualizer
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
from densepose.vis.extractor import CompoundExtractor, create_extractor
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor


def initialize_model(
    config_path: str = "/models/dense_pose/densepose_rcnn_R_50_FPN_s1x.yaml",
    model_path: str = "/models/dense_pose/densepose_rcnn_R_50_FPN_s1x.pkl",
    device="cuda",
    min_score=0.5,
):
    """
    Initialize the DensePose segmentation model.

    Args:
        config_path (str): Path to model configuration file
        model_path (str): Path to model weights file
        device (str): Device to use (default: "cuda")
        min_score (float): Minimum score threshold (default: 0.8)

    Returns:
        DefaultPredictor: DensePose predictor instance
        DensePoseResultsFineSegmentationVisualizer: DensePose visualizer instance
        CompoundExtractor: DensePose extractor instance
    """
    # Setup configuration
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min_score
    cfg.freeze()

    # Initialize predictor
    predictor = DefaultPredictor(cfg)

    # Create visualization pipeline
    vis = DensePoseResultsFineSegmentationVisualizer(cfg=cfg)
    visualizer = CompoundVisualizer([vis])
    extractor = create_extractor(vis)
    extractor = CompoundExtractor([extractor])

    return predictor, visualizer, extractor


def get_pose_mask(img, predictor: DefaultPredictor, visualizer: CompoundVisualizer, extractor: CompoundExtractor):
    """
    Process an image and return the segmentation mask.

    Args:
        img (numpy.ndarray): Input image in BGR format
        predictor: DensePose predictor instance
        visualizer: DensePose visualizer instance
        extractor: DensePose extractor instance

    Returns:
        numpy.ndarray: Processed image with segmentation mask
    """
    with torch.no_grad():
        outputs = predictor(img)
        # print(outputs)
        outputs = outputs["instances"]

    # Convert to grayscale and prepare for visualization
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = np.tile(image[:, :, np.newaxis], [1, 1, 3])

    # Extract data using same pattern as original script
    extracted_data = extractor(outputs)

    try:
        image_vis = visualizer.visualize(np.zeros_like(image), extracted_data)
        return image_vis
    except Exception as e:
        raise ValueError(f"Extracted data type: {e} | {type(extracted_data)} | Data length: {len(extracted_data)}")


# Example usage:
if __name__ == "__main__":
    # Initialize model (do this once)
    predictor, visualizer, extractor = initialize_model()

    # Read and process an image
    image_path = "test.jpg"
    img = read_image(image_path, format="BGR")
    result = get_pose_mask(img, predictor, visualizer, extractor)
    cv2.imwrite("output.png", result)

    # Example with webcam (uncomment to use)
    """
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = get_pose_mask(frame, predictor, visualizer, extractor)
        cv2.imshow('DensePose Segmentation', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    """

    print("Processing complete")
