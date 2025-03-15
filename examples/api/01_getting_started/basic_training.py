# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Getting Started with Anomalib Training using the Python API.

This example shows the basic steps to train an anomaly detection model
using the Anomalib Python API.
"""

# 1. Import required modules
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd

# 2. Create a dataset
# MVTecAD is a popular dataset for anomaly detection
datamodule = MVTecAD(
    root="./datasets/MVTecAD",  # Path to download/store the dataset
    category="bottle",  # MVTec category to use
    train_batch_size=32,  # Number of images per training batch
    eval_batch_size=32,  # Number of images per validation/test batch
    num_workers=8,  # Number of parallel processes for data loading
)

# 3. Initialize the model
# EfficientAd is a good default choice for beginners
model = EfficientAd()

# 4. Create the training engine
engine = Engine(max_epochs=10)  # Train for 10 epochs

# 5. Train the model
engine.fit(datamodule=datamodule, model=model)
