#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Advanced Custom Components with Anomalib CLI
# ---------------------------------------
# This example shows how to use custom metrics and evaluators.

# 1. Basic Metrics Setup
# Create metrics with specific fields to compute
echo "Training with basic metrics setup..."
anomalib train \
    --model efficient_ad \
    --model.evaluator.test_metrics auroc f1_score \
    --model.evaluator.val_metrics auroc f1_score \
    --model.evaluator.metrics.auroc.fields pred_score gt_label \
    --model.evaluator.metrics.f1_score.fields pred_label gt_label \
    --trainer.default_root_dir results/basic_metrics

# 2. Advanced Metrics Setup
# Create a comprehensive set of metrics
echo -e "\nTraining with comprehensive metrics..."
anomalib train \
    --model efficient_ad \
    --model.evaluator.test_metrics auroc f1_score precision recall \
    --model.evaluator.val_metrics auroc f1_score precision recall \
    --model.evaluator.metrics.auroc.fields pred_score gt_label \
    --model.evaluator.metrics.f1_score.fields pred_label gt_label \
    --model.evaluator.metrics.precision.fields pred_label gt_label \
    --model.evaluator.metrics.recall.fields pred_label gt_label \
    --model.evaluator.compute_on_cpu true \
    --trainer.default_root_dir results/advanced_metrics

# 3. Complete Training Pipeline with Custom Metrics
# Initialize components and run training
echo -e "\nRunning complete training pipeline..."
anomalib train \
    --model efficient_ad \
    --model.teacher_out_channels 384 \
    --data.category bottle \
    --data.train_batch_size 32 \
    --data.eval_batch_size 32 \
    --data.num_workers 8 \
    --model.evaluator.test_metrics auroc f1_score precision recall \
    --model.evaluator.val_metrics auroc f1_score precision recall \
    --model.evaluator.compute_on_cpu true \
    --trainer.max_epochs 20 \
    --trainer.accelerator auto \
    --trainer.devices 1 \
    --trainer.gradient_clip_val 0.1 \
    --trainer.enable_checkpointing true \
    --trainer.default_root_dir results/complete
