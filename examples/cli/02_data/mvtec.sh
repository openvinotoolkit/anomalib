#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Using MVTecAD Dataset with Anomalib CLI
# -----------------------------------
# This example shows different ways to use the MVTecAD dataset.

# 1. Basic Usage
# Train on a specific MVTecAD category
echo "Training on MVTecAD bottle category..."
anomalib train \
    --model efficient_ad \
    --data.category bottle

# 2. Advanced Configuration
# Customize data loading and preprocessing
echo -e "\nTraining with custom data settings..."
anomalib train \
    --model efficient_ad \
    --data.category bottle \
    --data.train_batch_size 32 \
    --data.eval_batch_size 32 \
    --data.num_workers 8 \
    --data.val_split_mode from_test \
    --data.val_split_ratio 0.5

# 3. Training Multiple Categories
# Train separate models for different categories
echo -e "\nTraining on multiple MVTecAD categories..."
for category in "bottle" "cable" "capsule"; do
    echo "Training on category: $category"
    anomalib train \
        --model efficient_ad \
        --data.category "$category" \
        --trainer.default_root_dir "results/$category"
done
