#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Using PaDiM Model with Anomalib CLI
# ---------------------------------
# This example shows how to use the PaDiM model for anomaly detection.

# 1. Basic Usage
# Train with default settings
echo "Training PaDiM with default settings..."
anomalib train \
    --model padim

# 2. Custom Configuration
# Train with custom model settings
echo -e "\nTraining with custom model settings..."
anomalib train \
    --model padim \
    --model.backbone resnet18 \
    --model.layers layer1 layer2 layer3 \
    --model.pre_trained true \
    --model.n_features 100

# 3. Advanced Training Pipeline
# Train with custom training settings
echo -e "\nTraining with custom pipeline settings..."
anomalib train \
    --model padim \
    --data.category bottle \
    --trainer.max_epochs 1 \
    --trainer.accelerator auto \
    --trainer.devices 1 \
    --trainer.default_root_dir results/padim

# 4. Feature Extraction Comparison
# Compare different backbones and feature combinations
echo -e "\nComparing different feature configurations..."
for backbone in "resnet18" "wide_resnet50_2"; do
    echo "Training with backbone: $backbone"
    anomalib train \
        --model padim \
        --model.backbone "$backbone" \
        --trainer.default_root_dir "results/padim_${backbone}"
done
