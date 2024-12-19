#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Using Patchcore Model with Anomalib CLI
# ------------------------------------
# This example shows how to use the Patchcore model for anomaly detection.

# 1. Basic Usage
# Train with default settings
echo "Training Patchcore with default settings..."
anomalib train \
    --model patchcore

# 2. Custom Configuration
# Train with custom model settings
echo -e "\nTraining with custom model settings..."
anomalib train \
    --model patchcore \
    --model.backbone wide_resnet50_2 \
    --model.layers layer2 layer3 \
    --model.pre_trained true \
    --model.num_neighbors 9

# 3. Advanced Training Pipeline
# Train with custom training settings
echo -e "\nTraining with custom pipeline settings..."
anomalib train \
    --model patchcore \
    --data.category bottle \
    --trainer.max_epochs 1 \
    --trainer.accelerator auto \
    --trainer.devices 1 \
    --trainer.default_root_dir results/patchcore

# 4. Multi-GPU Training
# Train using multiple GPUs for faster feature extraction
echo -e "\nTraining with multiple GPUs..."
anomalib train \
    --model patchcore \
    --data.category bottle \
    --trainer.accelerator gpu \
    --trainer.devices 2 \
    --trainer.strategy ddp
