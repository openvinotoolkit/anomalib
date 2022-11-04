"""Utils to help compute inference statistics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader

from anomalib.deploy import OpenVINOInferencer, TorchInferencer
from anomalib.models.components import AnomalyModule


class MockImageLoader:
    """Create mock images for inference on CPU based on the specifics of the original torch test dataset.

    Uses yield so as to avoid storing everything in the memory.

    Args:
        image_size (List[int]): Size of input image
        total_count (int): Total images in the test dataset
    """

    def __init__(self, image_size: List[int], total_count: int):
        self.total_count = total_count
        self.image_size = image_size
        self.image = np.ones((*self.image_size, 3)).astype(np.uint8)

    def __len__(self):
        """Get total count of images."""
        return self.total_count

    def __call__(self) -> Iterable[np.ndarray]:
        """Yield batch of generated images.

        Args:
            idx (int): Unused
        """
        for _ in range(self.total_count):
            yield self.image


def get_torch_throughput(
    config: Union[DictConfig, ListConfig], model: AnomalyModule, test_dataset: DataLoader
) -> float:
    """Tests the model on dummy data. Images are passed sequentially to make the comparision with OpenVINO model fair.

    Args:
        config (Union[DictConfig, ListConfig]): Model config.
        model (Path): Model on which inference is called.
        test_dataset (DataLoader): The test dataset used as a reference for the mock dataset.

    Returns:
        float: Inference throughput
    """
    torch.set_grad_enabled(False)
    model.eval()
    inferencer = TorchInferencer(config, model)
    torch_dataloader = MockImageLoader(config.dataset.image_size, len(test_dataset))
    start_time = time.time()
    # Since we don't care about performance metrics and just the throughput, use mock data.
    for image in torch_dataloader():
        inferencer.predict(image)

    # get throughput
    inference_time = time.time() - start_time
    throughput = len(test_dataset) / inference_time

    torch.set_grad_enabled(True)
    return throughput


def get_openvino_throughput(config: Union[DictConfig, ListConfig], model_path: Path, test_dataset: DataLoader) -> float:
    """Runs the generated OpenVINO model on a dummy dataset to get throughput.

    Args:
        config (Union[DictConfig, ListConfig]): Model config.
        model_path (Path): Path to folder containing the OpenVINO models. It then searches `model.xml` in the folder.
        test_dataset (DataLoader): The test dataset used as a reference for the mock dataset.

    Returns:
        float: Inference throughput
    """
    inferencer = OpenVINOInferencer(
        config, model_path / "openvino" / "model.xml", model_path / "openvino" / "meta_data.json"
    )
    openvino_dataloader = MockImageLoader(config.dataset.image_size, total_count=len(test_dataset))
    start_time = time.time()
    # Create test images on CPU. Since we don't care about performance metrics and just the throughput, use mock data.
    for image in openvino_dataloader():
        inferencer.predict(image)

    # get throughput
    inference_time = time.time() - start_time
    throughput = len(test_dataset) / inference_time

    return throughput
