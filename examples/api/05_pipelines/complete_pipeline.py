# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Complete Pipeline Example for Anomalib.

This example demonstrates a complete workflow including:
1. Training a model
2. Exporting for deployment
3. Running inference
"""

from pathlib import Path

from anomalib.data import MVTec, PredictDataset
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore

# 1. Training Phase
# ----------------
print("Starting Training Phase...")

# Initialize components
model = Patchcore()
datamodule = MVTec(
    root=Path("./datasets/MVTec"),
    category="bottle",
    train_batch_size=32,
)

# Configure training engine
engine = Engine(
    max_epochs=1,
    enable_checkpointing=True,
    default_root_dir="results",
)

# Train the model
engine.fit(model=model, datamodule=datamodule)

# 2. Export Phase
# --------------
print("\nStarting Export Phase...")

# Export to ONNX format
export_root = Path("exported_models")
export_root.mkdir(exist_ok=True)

export_to_onnx(
    model=model,
    export_root=export_root,
    input_size=(256, 256),  # Adjust based on your needs
    export_type=ExportType.TORCH,  # or OPENVINO
)

# 3. Inference Phase
# ----------------
print("\nStarting Inference Phase...")

# Prepare test data
test_data = PredictDataset(
    path=Path("path/to/test/images"),
    image_size=(256, 256),
)

# Run inference
predictions = engine.predict(
    model=model,
    dataset=test_data,
)

# Process results
print("\nProcessing Results...")
for prediction in predictions:
    image_path = prediction.image_path
    anomaly_score = prediction.pred_score
    is_anomalous = prediction.pred_label > 0.5

    print(f"Image: {image_path}")
    print(f"Anomaly Score: {anomaly_score:.3f}")
    print(f"Is Anomalous: {is_anomalous}\n")
