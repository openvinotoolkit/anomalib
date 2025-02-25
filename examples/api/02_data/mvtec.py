# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Example showing how to use the MVTecAD dataset with Anomalib.

MVTecAD is a widely-used dataset for anomaly detection, containing multiple
categories of industrial objects with various types of defects.
"""

from anomalib.data import MVTecAD

# 1. Basic Usage
# Load a specific category with default settings
datamodule = MVTecAD(
    root="./datasets/MVTecAD",
    category="bottle",
)

# 2. Advanced Configuration
# Customize data loading and preprocessing
datamodule = MVTecAD(
    root="./datasets/MVTecAD",
    category="bottle",
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=8,
    val_split_mode="from_test",  # Create validation set from test set
    val_split_ratio=0.5,  # Use 50% of test set for validation
)

# 3. Using Multiple Categories
# Train on multiple categories (if supported by the model)
for category in ["bottle", "cable", "capsule"]:
    category_data = MVTecAD(
        root="./datasets/MVTecAD",
        category=category,
    )
    # Use category_data with your model...
