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
    --model Patchcore \
    --data MVTecAD \
    --data.category bottle \
    --trainer.max_epochs 1 \
    --trainer.enable_checkpointing true \
    --trainer.default_root_dir results

# Checkpoint written by the training phase above
CKPT_PATH="results/Patchcore/MVTecAD/bottle/latest/weights/lightning/model.ckpt"

# 2. Export Phase
# --------------
echo -e "\nStarting Export Phase..."
anomalib export \
    --model Patchcore \
    --ckpt_path "$CKPT_PATH" \
    --export_root exported_models \
    --export_type onnx \
    --input_size "[256,256]"

# 3. Inference Phase
# ----------------
echo -e "\nStarting Inference Phase..."

# Run inference with the trained Lightning checkpoint
echo "Running inference with PyTorch model..."
anomalib predict \
    --model Patchcore \
    --ckpt_path "$CKPT_PATH" \
    --data path/to/test/images

# 4. Results Summary
# ----------------
echo -e "\nPipeline Complete!"
echo "Results are saved in:"
echo "- Training results: results/"
echo "- Exported models: exported_models/"
echo "- Predictions: predictions/"
