#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Advanced Logging with Anomalib CLI
# -------------------------------
# This example shows how to use different logging options.

# 1. Basic TensorBoard Logging
echo "Training with TensorBoard logging..."
anomalib train \
    --model patchcore \
    --trainer.logger tensorboard \
    --trainer.default_root_dir logs/tensorboard

# 2. Weights & Biases (WandB) Logging
echo -e "\nTraining with WandB logging..."
anomalib train \
    --model patchcore \
    --trainer.logger wandb \
    --trainer.logger.project anomalib \
    --trainer.logger.name patchcore_experiment \
    --trainer.default_root_dir logs/wandb

# 3. MLflow Logging
echo -e "\nTraining with MLflow logging..."
anomalib train \
    --model patchcore \
    --trainer.logger mlflow \
    --trainer.logger.experiment_name anomalib \
    --trainer.logger.tracking_uri logs/mlflow

# 4. Advanced Logging Configuration
echo -e "\nTraining with advanced logging settings..."
anomalib train \
    --model patchcore \
    --trainer.logger tensorboard \
    --trainer.logger.save_dir logs \
    --trainer.enable_checkpointing true \
    --trainer.log_every_n_steps 10 \
    --trainer.default_root_dir results

# 5. Logging with Model Export
echo -e "\nTraining with logging and model export..."
anomalib train \
    --model patchcore \
    --trainer.logger tensorboard \
    --trainer.default_root_dir results \
    --trainer.enable_checkpointing true \
    --export.format onnx \
    --export.export_root exported_models
