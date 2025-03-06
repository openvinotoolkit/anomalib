"""Example showing how to use the MVTec AD 2 dataset with Anomalib.

This example demonstrates how to:
1. Load and visualize the MVTec AD 2 dataset
2. Create a datamodule and use it for training
3. Access different test sets (public, private, mixed)
4. Work with custom transforms and visualization
"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage

from anomalib.data import MVTecAD2
from anomalib.data.datasets.base.image import ImageItem
from anomalib.data.datasets.image.mvtecad2 import MVTecAD2Dataset, TestType
from anomalib.data.utils import Split

# 1. Basic Usage
print("1. Basic Usage")
datamodule = MVTecAD2(
    root="./datasets/MVTec_AD_2",
    category="sheet_metal",
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=8,
)
datamodule.setup()  # This will prepare the dataset

# Print some information about the splits
print(f"Number of training samples: {len(datamodule.train_data)}")
print(f"Number of validation samples: {len(datamodule.val_data)}")
print(f"Number of test samples (public): {len(datamodule.test_public_data)}")
print(f"Number of test samples (private): {len(datamodule.test_private_data)}")
print(f"Number of test samples (private mixed): {len(datamodule.test_private_mixed_data)}")

# 2. Custom Transforms
print("\n2. Custom Transforms")
transform = Compose([
    ToImage(),
    Resize((256, 256)),
    ToDtype(torch.float32, scale=True),
])

# Create dataset with custom transform
datamodule = MVTecAD2(
    root="./datasets/MVTec_AD_2",
    category="sheet_metal",
    train_augmentations=transform,
    val_augmentations=transform,
    test_augmentations=transform,
)
datamodule.setup()

# 3. Different Test Sets
print("\n3. Accessing Different Test Sets")

# Get loaders for each test set
public_loader = datamodule.test_dataloader(test_type=TestType.PUBLIC)
private_loader = datamodule.test_dataloader(test_type=TestType.PRIVATE)
mixed_loader = datamodule.test_dataloader(test_type=TestType.PRIVATE_MIXED)

# Get sample batches
public_batch = next(iter(public_loader))
private_batch = next(iter(private_loader))
mixed_batch = next(iter(mixed_loader))

print("Public test batch shape:", public_batch.image.shape)
print("Private test batch shape:", private_batch.image.shape)
print("Private mixed test batch shape:", mixed_batch.image.shape)

# 4. Advanced Usage - Direct Dataset Access
print("\n4. Advanced Usage")

# Create datasets for each split
train_dataset = MVTecAD2Dataset(
    root="./datasets/MVTec_AD_2",
    category="sheet_metal",
    split=Split.TRAIN,
    augmentations=transform,
)

test_dataset = MVTecAD2Dataset(
    root="./datasets/MVTec_AD_2",
    category="sheet_metal",
    split=Split.TEST,
    test_type=TestType.PUBLIC,  # Use public test set
    augmentations=transform,
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=train_dataset.collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=test_dataset.collate_fn)

# Get some sample images
train_samples = next(iter(train_loader))
test_samples = next(iter(test_loader))

print("Train Dataset:")
print(f"- Number of samples: {len(train_dataset)}")
print(f"- Image shape: {train_samples.image.shape}")
print(f"- Labels: {train_samples.gt_label}")

print("\nTest Dataset:")
print(f"- Number of samples: {len(test_dataset)}")
print(f"- Image shape: {test_samples.image.shape}")
print(f"- Labels: {test_samples.gt_label}")
if hasattr(test_samples, "gt_mask") and test_samples.gt_mask is not None:
    print(f"- Mask shape: {test_samples.gt_mask.shape}")


# 5. Visualize some samples
def show_samples(samples: ImageItem, title: str) -> None:
    """Helper function to display samples."""
    if samples.image is None or samples.gt_label is None:
        msg = "Samples must have image and label data"
        raise ValueError(msg)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle(title)

    for i in range(4):
        img = samples.image[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis("off")
        if hasattr(samples, "gt_mask") and samples.gt_mask is not None:
            mask = samples.gt_mask[i].squeeze().numpy()
            axes[i].imshow(mask, alpha=0.3, cmap="Reds")
        label = "Normal" if samples.gt_label[i] == 0 else "Anomaly"
        axes[i].set_title(label)

    plt.tight_layout()
    plt.show()


# Show training samples (normal only)
show_samples(train_samples, "Training Samples (Normal)")

# Show test samples (mix of normal and anomalous)
show_samples(test_samples, "Test Samples (Normal + Anomalous)")

if __name__ == "__main__":
    print("\nMVTec AD 2 Dataset example completed successfully!")
