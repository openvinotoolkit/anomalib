"""Example showing how to use the MVTec LOCO dataset with Anomalib.

MVTec LOCO is a dataset for detecting logical and structural anomalies in images.
It contains 5 categories of industrial objects with various types of defects.
"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.data import MVTecLOCO

# 1. Basic Usage
# Load a specific category with default settings
datamodule = MVTecLOCO(
    root="./datasets/MVTec_LOCO",
    category="breakfast_box",
)
datamodule.prepare_data()
datamodule.setup()
i, data = next(enumerate(datamodule.test_dataloader()))


# 2. Advanced Configuration
# Customize data loading and preprocessing
datamodule = MVTecLOCO(
    root="./datasets/MVTec_LOCO",
    category="juice_bottle",
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=8,
    val_split_mode="from_test",  # Create validation set from test set
    val_split_ratio=0.5,  # Use 50% of test set for validation
)

# 3. Using Multiple Categories
# Train on multiple categories (if supported by the model)
for category in ["breakfast_box", "juice_bottle", "pushpins"]:
    category_data = MVTecLOCO(
        root="./datasets/MVTec_LOCO",
        category=category,
    )
    # Use category_data with your model...

# 4. Accessing Dataset Properties
# Get information about the dataset
print(f"Number of training samples: {len(datamodule.train_data)}")
print(f"Number of validation samples: {len(datamodule.val_data)}")
print(f"Number of test samples: {len(datamodule.test_data)}")

# 5. Working with Data Samples
# Get a sample from the dataset
sample = datamodule.train_data[0]
print("\nSample keys:", list(sample.__dict__.keys()))
print("Image shape:", sample.image.shape if sample.image is not None else None)
print("Mask shape:", sample.gt_mask.shape if sample.gt_mask is not None else None)
print("Label:", sample.gt_label)

# 6. Using with a Model
# Example of using the datamodule with a model
from anomalib.engine import Engine  # noqa: E402
from anomalib.models import Patchcore  # noqa: E402

# Initialize model
model = Patchcore(backbone="wide_resnet50_2", layers=["layer3"], coreset_sampling_ratio=0.1)

# Train using the Engine
engine = Engine()
engine.fit(model=model, datamodule=datamodule)

# Get predictions
predictions = engine.predict(model=model, datamodule=datamodule)
