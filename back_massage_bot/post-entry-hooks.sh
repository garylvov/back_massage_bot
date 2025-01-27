#!/usr/bin/env bash
# If you modify me, your next docker build may take long!
# Check if we're in the right environment, if not activate it
if [[ "$CONDA_DEFAULT_ENV" != "back_massage_bot" ]]; then
    conda activate back_massage_bot
fi

# RUN pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose
# RUN wget -O /back_massage_bot/models/dense_pose/densepose_rcnn_R_50_FPN_soft_s1x.pkl \
# https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_s1x/250533982/model_final_2c4512.pkl
# RUN wget -O /back_massage_bot/models/dense_pose/densepose_rcnn_R_50_FPN_soft_s1x.yaml \
# https://github.com/facebookresearch/detectron2/blob/9604f5995cc628619f0e4fd913453b4d7d61db3f/projects/DensePose/configs/cse/densepose_rcnn_R_50_FPN_soft_s1x.yaml

echo "Activated Conda."
