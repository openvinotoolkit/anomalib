#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Getting Started with Anomalib Training
# ------------------------------------
# This example shows the basic steps to train an anomaly detection model.

# 1. Basic Training
# Train a model using default configuration (recommended for beginners)
echo "Training with default configuration..."
anomalib train --model efficient_ad

# 2. Training with Basic Customization
# Customize basic parameters like batch size and epochs
echo -e "\nTraining with custom parameters..."
anomalib train --model efficient_ad \
    --data.train_batch_size 32 \
    --trainer.max_epochs 10

# 3. Using a Different Dataset
# Train on a specific category of MVTecAD dataset
echo -e "\nTraining on MVTecAD bottle category..."
anomalib train --model efficient_ad \
    --data.category bottle
