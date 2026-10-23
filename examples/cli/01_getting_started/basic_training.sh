#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Getting Started with Anomalib Training
# ------------------------------------
# This example shows the basic steps to train an anomaly detection model.

# 1. Basic Training
# Train and test a model using default configuration on MVTecAD bottle (default category)
echo "Training with default configuration..."
anomalib train --model Patchcore --data anomalib.data.MVTecAD

# 2. Training with Basic Customization
# Customize basic parameters like batch size and epochs.
# For example `EfficientAd` requires a train batch size of 1
echo -e "\nTraining with custom parameters..."
anomalib train --model EfficientAd --data anomalib.data.MVTecAD \
    --data.category hazelnut \
    --data.train_batch_size 1 \
    --trainer.max_epochs 200

# 3. Train with config file
# Train with a custom config file
echo -e "\nTraining with config file..."
anomalib train --config path/to/config.yaml

# 4. Export a trained model into OpenVINO
echo -e "\nExporting model into OpenVINO..."
anomalib export --model Patchcore \
        --ckpt_path /path/to/model.ckpt \
        --export_type openvino
