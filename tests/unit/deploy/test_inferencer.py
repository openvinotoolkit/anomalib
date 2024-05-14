"""Tests for Torch and OpenVINO inferencers."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pytest
import torch

from anomalib import TaskType
from anomalib.deploy import ExportType, OpenVINOInferencer, TorchInferencer
from anomalib.engine import Engine
from anomalib.models import Padim


class _MockImageLoader:
    """Create mock images for inference on CPU based on the specifics of the original torch test dataset.

    Uses yield so as to avoid storing everything in the memory.

    Args:
        image_size (list[int]): Size of input image
        total_count (int): Total images in the test dataset
    """

    def __init__(self, image_size: list[int], total_count: int, as_numpy: bool = False) -> None:
        self.total_count = total_count
        self.image_size = image_size
        if as_numpy:
            self.image = np.ones((*self.image_size, 3)).astype(np.uint8)
        else:
            self.image = torch.rand((3, *self.image_size))

    def __len__(self) -> int:
        """Get total count of images."""
        return self.total_count

    def __call__(self) -> Iterable[np.ndarray] | Iterable[torch.Tensor]:
        """Yield batch of generated images.

        Args:
            idx (int): Unused
        """
        for _ in range(self.total_count):
            yield self.image


@pytest.mark.parametrize(
    "task",
    [
        TaskType.CLASSIFICATION,
        TaskType.DETECTION,
        TaskType.SEGMENTATION,
    ],
)
def test_torch_inference(task: TaskType, ckpt_path: Callable[[str], Path]) -> None:
    """Tests Torch inference.

    Model is not trained as this checks that the inferencers are working.

    Args:
        task (TaskType): Task type
        ckpt_path: Callable[[str], Path]: Path to trained PADIM model checkpoint.
        dataset_path (Path): Path to dummy dataset.
    """
    model = Padim()
    engine = Engine(task=task)
    export_root = ckpt_path("Padim").parent.parent
    engine.export(
        model=model,
        export_type=ExportType.TORCH,
        export_root=export_root,
        ckpt_path=str(ckpt_path("Padim")),
    )
    # Test torch inferencer
    torch_inferencer = TorchInferencer(
        path=export_root / "weights" / "torch" / "model.pt",
        device="cpu",
    )
    torch_dataloader = _MockImageLoader([256, 256], total_count=1)
    with torch.no_grad():
        for image in torch_dataloader():
            prediction = torch_inferencer.predict(image)
            assert 0.0 <= prediction.pred_score <= 1.0  # confirm if predicted scores are normalized


@pytest.mark.parametrize(
    "task",
    [
        TaskType.CLASSIFICATION,
        TaskType.DETECTION,
        TaskType.SEGMENTATION,
    ],
)
def test_openvino_inference(task: TaskType, ckpt_path: Callable[[str], Path]) -> None:
    """Tests OpenVINO inference.

    Model is not trained as this checks that the inferencers are working.

    Args:
        task (TaskType): Task type
        ckpt_path: Callable[[str], Path]: Path to trained PADIM model checkpoint.
        dataset_path (Path): Path to dummy dataset.
    """
    model = Padim()
    engine = Engine(task=task)
    export_dir = ckpt_path("Padim").parent.parent
    exported_xml_file_path = engine.export(
        model=model,
        export_type=ExportType.OPENVINO,
        export_root=export_dir,
        ckpt_path=str(ckpt_path("Padim")),
    )

    # Test OpenVINO inferencer
    openvino_inferencer = OpenVINOInferencer(
        exported_xml_file_path,
        exported_xml_file_path.parent / "metadata.json",
    )
    openvino_dataloader = _MockImageLoader([256, 256], total_count=1, as_numpy=True)
    for image in openvino_dataloader():
        prediction = openvino_inferencer.predict(image)
        assert 0.0 <= prediction.pred_score <= 1.0  # confirm if predicted scores are normalized
