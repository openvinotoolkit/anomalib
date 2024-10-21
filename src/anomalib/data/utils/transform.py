"""Utility functions for data transforms."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Transform

from anomalib.data import AnomalibDataModule


def set_datamodule_transform(datamodule: AnomalibDataModule, transform: Transform, stage: str) -> None:
    """Set a transform for a specific stage in a AnomalibDataModule.

    This function allows you to set a custom transform for a specific stage (train, val, or test)
    in an AnomalibDataModule. It checks if the datamodule has the corresponding dataset attribute
    and if that dataset has a transform attribute, then sets the new transform.

    Args:
        datamodule: The AnomalibDataModule to set the transform for.
        transform: The transform to set.
        stage: The stage (e.g., 'train', 'val', 'test') to set the transform for.

    Examples:
        >>> from torchvision.transforms.v2 import Compose, Resize, ToTensor
        >>> from anomalib.data import MVTec
        >>> from anomalib.data.utils.transform import set_datamodule_transform

        >>> # Create a datamodule and check its transform
        >>> datamodule = MVTec(root="path/to/dataset", category="bottle")
        >>> datamodule.setup()
        >>> print(datamodule.train_data.transform)  # Output: None or default transform

        >>> # Define a custom transform and set it for the training stage
        >>> custom_transform = Compose([Resize((224, 224)), ToTensor()])
        >>> set_datamodule_transform(datamodule, custom_transform, "train")
        >>> print(datamodule.train_data.transform)  # Output: Compose([Resize((224, 224)), ToTensor()])

        >>> # You can also set transforms for validation and test stages
        >>> set_datamodule_transform(datamodule, custom_transform, "val")
        >>> set_datamodule_transform(datamodule, custom_transform, "test")

        >>> # The dataloaders will now use the custom transforms
        >>> train_dataloader = datamodule.train_dataloader()
        >>> val_dataloader = datamodule.val_dataloader()
        >>> test_dataloader = datamodule.test_dataloader()
    """
    dataset_attr = f"{stage}_data"
    if hasattr(datamodule, dataset_attr):
        dataset = getattr(datamodule, dataset_attr)
        if hasattr(dataset, "transform"):
            dataset.transform = transform


def set_dataloader_transform(dataloader: DataLoader | Sequence[DataLoader], transform: Transform) -> None:
    """Set a transform for a dataloader or list of dataloaders.

    Args:
        dataloader: The dataloader(s) to set the transform for. Can be a single DataLoader,
                    a callable returning a DataLoader, or a list of DataLoaders.
        transform: The transform to set.

    Examples:
        >>> from torch.utils.data import DataLoader
        >>> from torchvision.transforms.v2 import Compose, Resize, ToTensor
        >>> from anomalib.data import MVTecDataset
        >>> from anomalib.data.utils.transform import set_dataloader_transform

        >>> # Create a dataset and dataloader
        >>> dataset = MVTecDataset(root="./datasets/MVTec", category="bottle", task="segmentation")
        >>> dataloader = DataLoader(dataset, batch_size=32)

        >>> # Define a custom transform and set it for a single DataLoader
        >>> custom_transform = Compose([Resize((224, 224)), ToTensor()])
        >>> set_dataloader_transform(dataloader, custom_transform)
        >>> print(dataloader.dataset.transform)  # Output: Compose([Resize((224, 224)), ToTensor()])

        >>> # Set the transform for a list of DataLoaders
        >>> dataset_bottle = MVTecDataset(root="./datasets/MVTec", category="bottle", task="segmentation")
        >>> dataset_cable = MVTecDataset(root="./datasets/MVTec", category="cable", task="segmentation")
        >>> dataloader_list = [
        ...     DataLoader(dataset_bottle, batch_size=32),
        ...     DataLoader(dataset_cable, batch_size=32)
        ... ]
        >>> set_dataloader_transform(dataloader_list, custom_transform)
        >>> for dl in dataloader_list:
        ...     print(dl.dataset.transform)  # Output: Compose([Resize((224, 224)), ToTensor()])
    """
    if isinstance(dataloader, DataLoader):
        if hasattr(dataloader.dataset, "transform"):
            dataloader.dataset.transform = transform
    elif isinstance(dataloader, Sequence):
        for dl in dataloader:
            set_dataloader_transform(dl, transform)
    else:
        msg = f"Unsupported dataloader type: {type(dataloader)}"
        raise TypeError(msg)
