# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for the benchmark pipeline."""

from typing import Any

import torch
from torch.utils.data import DataLoader

from anomalib.data import AnomalibDataModule
from anomalib.models import AnomalibModule


def get_test_dataloader(
    datamodule: AnomalibDataModule,
    batch_size: int | None = None,
) -> DataLoader | None:
    """Get the test dataloader from the datamodule, optionally with a custom batch size.

    Args:
        datamodule: Data module providing the dataset.
        batch_size: Optional override for dataloader batch size.

    Returns:
        DataLoader | None: The test dataloader if available, otherwise None.
    """
    try:
        dataloaders = datamodule.test_dataloader()
        dataloader = dataloaders[0] if isinstance(dataloaders, (list, tuple)) else dataloaders
        if not hasattr(dataloader, "dataset"):
            return None

        if batch_size is not None and getattr(dataloader, "batch_size", None) != batch_size:
            # Re-create a DataLoader with a different batch size while preserving common settings.
            kwargs: dict[str, Any] = {
                "dataset": dataloader.dataset,
                "batch_size": batch_size,
                "shuffle": False,
                "num_workers": getattr(dataloader, "num_workers", 0),
                "pin_memory": getattr(dataloader, "pin_memory", False),
                "drop_last": getattr(dataloader, "drop_last", False),
                "collate_fn": getattr(dataloader, "collate_fn", None),
            }
            for key in ("timeout", "worker_init_fn", "prefetch_factor", "persistent_workers"):
                if hasattr(dataloader, key):
                    kwargs[key] = getattr(dataloader, key)
            dataloader = DataLoader(**kwargs)
        
        return dataloader
    except Exception:  # noqa: BLE001
        return None


def get_device_from_model(model: AnomalibModule) -> torch.device:
    """Get the device of the model."""
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:  # noqa: BLE001
            device = torch.device("cpu")
    return device


def extract_images_from_batch(batch: Any) -> torch.Tensor | None:  # noqa: ANN401
    """Extract images from a batch, supporting different batch structures."""
    if hasattr(batch, "image"):
        return batch.image
    if isinstance(batch, dict) and "image" in batch:
        return batch["image"]
    return None
