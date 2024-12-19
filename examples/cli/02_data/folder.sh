#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Using Custom Folder Dataset with Anomalib CLI
# ------------------------------------------
# This example shows how to use your own dataset organized in folders.

# 1. Basic Usage with Default Structure
# Train using the default folder structure
echo "Training on custom dataset with default structure..."
anomalib train \
    --model efficient_ad \
    --data Folder \
    --data.name my_dataset \
    --data.root ./datasets/my_dataset \
    --data.normal_dir good \
    --data.abnormal_dir defect

# 2. Custom Configuration
# Train with custom data settings
echo -e "\nTraining with custom data settings..."
anomalib train \
    --model efficient_ad \
    --data Folder \
    --data.name my_dataset \
    --data.root ./datasets/my_dataset \
    --data.normal_dir normal \
    --data.abnormal_dir anomaly \
    --data.train_batch_size 32 \
    --data.eval_batch_size 32 \
    --data.num_workers 8

# 3. Training with Multiple Dataset Variations
# Train on different subsets or configurations
echo -e "\nTraining on multiple dataset variations..."
for defect_type in "scratch" "crack" "stain"; do
    echo "Training on defect type: $defect_type"
    anomalib train \
        --model efficient_ad \
        --data Folder \
        --data.name my_dataset \
        --data.root "./datasets/my_dataset/$defect_type" \
        --data.normal_dir good \
        --data.abnormal_dir "$defect_type" \
        --trainer.default_root_dir "results/$defect_type"
done
