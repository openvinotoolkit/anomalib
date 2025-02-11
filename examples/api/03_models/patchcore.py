# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Example showing how to use the Patchcore model.

Patchcore is a memory-based model that uses a pretrained CNN backbone
to extract and store patch features for anomaly detection.
"""

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

# 1. Basic Usage
# Initialize with default settings
model = Patchcore()

# 2. Custom Configuration
# Configure model parameters
model = Patchcore(
    backbone="wide_resnet50_2",  # Feature extraction backbone
    layers=["layer2", "layer3"],  # Layers to extract features from
    pre_trained=True,  # Use pretrained weights
    num_neighbors=9,  # Number of nearest neighbors
)

# 3. Training Pipeline
# Set up the complete training pipeline
datamodule = MVTecAD(
    root="./datasets/MVTecAD",
    category="bottle",
    train_batch_size=32,
    eval_batch_size=32,  # Important for feature extraction
)

# Initialize training engine with specific settings
engine = Engine(
    max_epochs=1,  # Patchcore typically needs only one epoch
    accelerator="auto",  # Automatically detect GPU/CPU
    devices=1,  # Number of devices to use
)

# Train the model
engine.fit(
    model=model,
    datamodule=datamodule,
)
