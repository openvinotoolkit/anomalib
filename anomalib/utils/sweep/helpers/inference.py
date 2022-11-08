"""Utils to help compute inference statistics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from pathlib import Path
from typing import Union

import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader

from anomalib.deploy import OpenVINOInferencer, TorchInferencer
from anomalib.models.components import AnomalyModule


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

    device = config.trainer.accelerator
    if device == "gpu":
        device = "cuda"

    inferencer = TorchInferencer(config, model.to(device), device=device)
    start_time = time.time()
    for image_path in test_dataset.samples.image_path:
        inferencer.predict(image_path)

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
    start_time = time.time()
    for image_path in test_dataset.samples.image_path:
        inferencer.predict(image_path)

    # get throughput
    inference_time = time.time() - start_time
    throughput = len(test_dataset) / inference_time

    return throughput
