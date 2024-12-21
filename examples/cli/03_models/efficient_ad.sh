#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Using EfficientAd Model with Anomalib CLI
# --------------------------------------
# This example shows how to use the EfficientAd model for anomaly detection.

# 1. Basic Usage
# Train with default settings
echo "Training EfficientAd with default settings..."
anomalib train \
    --model efficient_ad

# 2. Custom Configuration
# Train with custom model settings
echo -e "\nTraining with custom model settings..."
anomalib train \
    --model efficient_ad \
    --model.teacher_out_channels 384 \
    --model.model_size m \
    --model.lr 1e-4

# 3. Advanced Training Pipeline
# Train with custom training settings
echo -e "\nTraining with custom pipeline settings..."
anomalib train \
    --model efficient_ad \
    --data.category bottle \
    --trainer.max_epochs 20 \
    --trainer.accelerator auto \
    --trainer.devices 1 \
    --trainer.default_root_dir results/efficient_ad

# 4. Hyperparameter Search
# Train multiple variations to find best settings
echo -e "\nRunning hyperparameter search..."
for channels in 128 256 384; do
    echo "Training with $channels channels..."
    anomalib train \
        --model efficient_ad \
        --model.teacher_out_channels $channels \
        --model.student_out_channels $channels \
        --trainer.default_root_dir "results/efficient_ad_${channels}"
done
