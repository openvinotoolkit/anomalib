# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Getting Started with Anomalib Inference using the Python API.

This example shows how to perform inference on a trained model
using the Anomalib Python API.
"""

# 1. Import required modules
from pathlib import Path

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import EfficientAd

# 2. Initialize the model and load weights
model = EfficientAd()
engine = Engine()

# 3. Prepare test data
# You can use a single image or a folder of images
dataset = PredictDataset(
    path=Path("path/to/test/images"),
    image_size=(256, 256),
)

# 4. Get predictions
predictions = engine.predict(
    model=model,
    dataset=dataset,
    ckpt_path="path/to/model.ckpt",
)

# 5. Access the results
if predictions is not None:
    for prediction in predictions:
        image_path = prediction.image_path
        anomaly_map = prediction.anomaly_map  # Pixel-level anomaly heatmap
        pred_label = prediction.pred_label  # Image-level label (0: normal, 1: anomalous)
        pred_score = prediction.pred_score  # Image-level anomaly score
