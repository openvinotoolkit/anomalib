#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Getting Started with Anomalib Inference
# ------------------------------------
# This example shows how to perform inference using a trained model.

# 1. Basic Inference
# Predict on a single image or folder using a trained model
echo "Running inference on test images..."
anomalib predict \
    --model efficient_ad \
    --weights path/to/model.ckpt \
    --input path/to/test/images

# 2. Inference with Visualization
# Save visualization of the predictions
echo -e "\nRunning inference with visualization..."
anomalib predict \
    --model efficient_ad \
    --weights path/to/model.ckpt \
    --input path/to/test/images \
    --visualization

# 3. Inference with Custom Settings
# Customize prediction threshold and output path
echo -e "\nRunning inference with custom settings..."
anomalib predict \
    --model efficient_ad \
    --weights path/to/model.ckpt \
    --input path/to/test/images \
    --output path/to/results \
    --threshold 0.5  # Custom anomaly threshold