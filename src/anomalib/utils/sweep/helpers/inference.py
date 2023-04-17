"""Utils to help compute inference statistics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

from anomalib.deploy import OpenVINOInferencer, TorchInferencer


def get_torch_throughput(model_path: str | Path, test_dataset: Dataset, device: str) -> float:
    """Tests the model on dummy data. Images are passed sequentially to make the comparision with OpenVINO model fair.

    Args:
        model_path (str, Path): Path to folder containing the Torch models.
        test_dataset (Dataset): The test dataset used as a reference for the mock dataset.
        device (str): Device to use for inference. Options are auto, cpu, gpu, cuda.

    Returns:
        float: Inference throughput
    """
    model_path = Path(model_path)
    torch.set_grad_enabled(False)

    if device == "gpu":
        device = "cuda"

    inferencer = TorchInferencer(
        path=model_path / "weights" / "torch" / "model.pt",
        device=device,
    )
    start_time = time.time()
    for image_path in test_dataset.samples.image_path:
        inferencer.predict(image_path)

    # get throughput
    inference_time = time.time() - start_time
    throughput = len(test_dataset) / inference_time

    torch.set_grad_enabled(True)
    return throughput


def get_openvino_throughput(model_path: str | Path, test_dataset: Dataset) -> float:
    """Runs the generated OpenVINO model on a dummy dataset to get throughput.

    Args:
        model_path (str, Path): Path to folder containing the OpenVINO models. It then searches `model.xml` in folder.
        test_dataset (Dataset): The test dataset used as a reference for the mock dataset.

    Returns:
        float: Inference throughput
    """
    model_path = Path(model_path)

    inferencer = OpenVINOInferencer(
        path=model_path / "weights" / "openvino" / "model.xml",
        metadata_path=model_path / "weights" / "openvino" / "metadata.json",
    )
    start_time = time.time()
    for image_path in test_dataset.samples.image_path:
        inferencer.predict(image_path)

    # get throughput
    inference_time = time.time() - start_time
    throughput = len(test_dataset) / inference_time

    return throughput
