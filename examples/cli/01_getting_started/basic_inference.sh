#!/usr/bin/env bash
# shellcheck shell=bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Getting Started with Anomalib Inference
# This example shows how to perform inference using Engine().predict() arguments.

echo "=== Anomalib Inference Examples ==="

echo -e "\n1. Basic Inference with Checkpoint Path"
echo "# Predict using a model checkpoint"
anomalib predict \
    --ckpt_path "./results/efficient_ad/mvtecad/bottle/weights/model.ckpt" \
    --data_path path/to/image.jpg

echo -e "\n2. Inference with Directory Path"
echo "# Predict on all images in a directory"
anomalib predict \
    --ckpt_path "./results/efficient_ad/mvtecad/bottle/weights/model.ckpt" \
    --data_path "./datasets/mvtecad/bottle/test"

echo -e "\n3. Inference with Datamodule"
echo "# Use a datamodule for inference"
anomalib predict \
    --ckpt_path "./results/my_dataset/weights/model.ckpt" \
    --datamodule.class_path anomalib.data.Folder \
    --datamodule.init_args.name "my_dataset" \
    --datamodule.init_args.root "./datasets/my_dataset" \
    --datamodule.init_args.normal_dir "good" \
    --datamodule.init_args.abnormal_dir "defect"

echo -e "\n4. Inference with Return Predictions"
echo "# Return predictions instead of saving to disk"
anomalib predict \
    --ckpt_path "./results/efficient_ad/mvtecad/bottle/weights/model.ckpt" \
    --data_path path/to/image.jpg \
    --return_predictions

echo -e "\n=== Example Output ==="
echo '
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
[2024-01-01 12:00:00][INFO][anomalib][predict]: Loading model from ./results/my_dataset/weights/model.ckpt
[2024-01-01 12:00:01][INFO][anomalib][predict]: Prediction started
[2024-01-01 12:00:02][INFO][anomalib][predict]: Predictions saved to ./results/my_dataset/predictions'

echo -e "\nNote: Replace paths according to your setup."
echo "The predictions will be saved in the results directory by default unless --return_predictions is used."
