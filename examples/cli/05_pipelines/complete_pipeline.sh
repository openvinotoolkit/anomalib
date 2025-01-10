#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Complete Anomalib Pipeline Example
# ------------------------------
# This script demonstrates a complete workflow from training to deployment.

# 0. Setup
# Create necessary directories
mkdir -p results exported_models predictions

# 1. Training Phase
# ----------------
echo "Starting Training Phase..."
anomalib train \
    --model patchcore \
    --data.category bottle \
    --trainer.max_epochs 1 \
    --trainer.enable_checkpointing true \
    --trainer.default_root_dir results

# 2. Export Phase
# --------------
echo -e "\nStarting Export Phase..."
anomalib export \
    --model patchcore \
    --weights results/*/checkpoints/*.ckpt \
    --export_root exported_models \
    --export_mode onnx \
    --input_size 256 256

# 3. Inference Phase
# ----------------
echo -e "\nStarting Inference Phase..."

# 3.1 Using PyTorch Model
echo "Running inference with PyTorch model..."
anomalib predict \
    --model patchcore \
    --weights results/*/checkpoints/*.ckpt \
    --input path/to/test/images \
    --output predictions/torch_results

# 3.2 Using Exported Model
echo -e "\nRunning inference with exported ONNX model..."
anomalib predict \
    --model patchcore \
    --weights exported_models/model.onnx \
    --input path/to/test/images \
    --output predictions/onnx_results

# 4. Results Summary
# ----------------
echo -e "\nPipeline Complete!"
echo "Results are saved in:"
echo "- Training results: results/"
echo "- Exported models: exported_models/"
echo "- Predictions: predictions/"
