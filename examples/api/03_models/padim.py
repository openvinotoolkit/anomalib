# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Example showing how to use the Padim model.

PaDiM (Patch Distribution Modeling) is a model that uses pretrained CNN features
and multivariate Gaussian modeling for anomaly detection.
"""

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim

# 1. Basic Usage
# Initialize with default settings
model = Padim()

# 2. Custom Configuration
# Configure model parameters
model = Padim(
    backbone="resnet18",  # Feature extraction backbone
    layers=["layer1", "layer2", "layer3"],  # Layers to extract features from
    pre_trained=True,  # Use pretrained weights
    n_features=100,  # Number of features to retain
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
    max_epochs=1,  # PaDiM needs only one epoch
    accelerator="auto",  # Automatically detect GPU/CPU
    devices=1,  # Number of devices to use
)

# Train the model
engine.fit(
    model=model,
    datamodule=datamodule,
)
