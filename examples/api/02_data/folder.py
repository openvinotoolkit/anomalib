# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Example showing how to use your own dataset with Anomalib.

This example demonstrates how to use a custom folder dataset where images
are organized in a specific directory structure.
"""

from pathlib import Path

from anomalib.data import Folder

# 1. Basic Usage with Default Structure
# Default structure expects:
# - train/good: Normal (good) training images
# - test/good: Normal test images
# - test/defect: Anomalous test images
datamodule = Folder(
    name="my_dataset",
    root=Path("./datasets/my_dataset"),
    normal_dir="good",  # Subfolder containing normal images
    abnormal_dir="defect",  # Subfolder containing anomalous images
)

# 2. Custom Directory Structure
# For a different directory structure:
# my_dataset/
# ├── train/
# │   └── normal/         # Normal training images
# ├── val/
# │   ├── normal/         # Normal validation images
# │   └── anomaly/        # Anomalous validation images
# └── test/
#     ├── normal/         # Normal test images
#     └── anomaly/        # Anomalous test images
datamodule = Folder(
    name="my_dataset",
    root=Path("./datasets/my_dataset"),
    normal_dir="normal",  # Subfolder containing normal images
    abnormal_dir="anomaly",  # Subfolder containing anomalous images
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=8,
)
