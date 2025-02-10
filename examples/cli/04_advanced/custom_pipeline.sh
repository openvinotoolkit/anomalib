#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Advanced Anomalib Pipeline Configuration
# -------------------------------------
# This example shows how to configure advanced pipeline settings using the CLI.

# 1. Training with Custom Components
# Configure pre-processing, metrics, and visualization
echo "Training with custom pipeline components..."
anomalib train \
    --model patchcore \
    --data MVTecAD \
    --data.category bottle \
    --model.backbone resnet18 \
    --model.layers layer2 layer3 \
    --pre_processor.transform.name Compose \
    --pre_processor.transform.transforms "[
        {name: Resize, size: [256, 256]},
        {name: ToTensor},
        {name: Normalize, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
    ]" \
    --metrics "[auroc, f1_score]" \

# 2. Advanced Training Configuration
# Configure training behavior and optimization
echo -e "\nTraining with advanced settings..."
anomalib train \
    --model patchcore \
    --data MVTecAD \
    --trainer.max_epochs 1 \
    --trainer.accelerator gpu \
    --trainer.devices 1 \
    --trainer.precision 16 \
    --trainer.deterministic true \
    --optimizer.name Adam \
    --optimizer.lr 0.001 \
    --scheduler.name CosineAnnealingLR \
    --scheduler.T_max 100

# 3. Export and Deploy
# Export the trained model and run inference
echo -e "\nExporting and running inference..."
# First, export the model
anomalib export \
    --model patchcore \
    --weights path/to/weights.ckpt \
    --export_mode onnx \
    --output_path exported_models

# Then, run inference with the exported model
anomalib predict \
    --model patchcore \
    --weights exported_models/model.onnx \
    --input path/to/test/images \
    --output results/predictions \

# 4. Hyperparameter Search
# Run multiple training configurations
echo -e "\nRunning hyperparameter search..."
for backbone in "resnet18" "wide_resnet50_2"; do
    for layer_combo in "layer2,layer3" "layer1,layer2,layer3"; do
        IFS=',' read -ra layers <<< "$layer_combo"
        echo "Training with backbone: $backbone, layers: ${layers[*]}"
        anomalib train \
            --model patchcore \
            --data MVTecAD \
            --model.backbone "$backbone" \
            --model.layers "${layers[@]}" \
            --trainer.default_root_dir "results/search/${backbone}_${layer_combo}"
    done
done
