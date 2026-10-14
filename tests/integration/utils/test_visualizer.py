# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the Visualizer class."""

from collections.abc import Callable
from pathlib import Path

from torch.utils.data import DataLoader

from anomalib.data import ImageBatch, MVTecAD, PredictDataset
from anomalib.engine import Engine
from anomalib.models import Padim


class TestVisualizer:
    """Test visualization callback for test and predict with different task types."""

    @staticmethod
    def test_model_visualizer_mode(
        ckpt_path: Callable[[str], Path],
        project_path: Path,
        mvtecad_path: Path,
    ) -> None:
        """Test combination of model/visualizer/mode on only 1 epoch as a sanity check before merge."""
        checkpoint_path: Path = ckpt_path("Padim")
        model = Padim(evaluator=False)
        engine = Engine(
            default_root_dir=project_path,
            fast_dev_run=True,
            devices=1,
        )
        datamodule = MVTecAD(root=mvtecad_path, category="dummy")
        engine.test(model=model, datamodule=datamodule, ckpt_path=str(checkpoint_path))

        dataset = PredictDataset(path=mvtecad_path / "dummy" / "test")
        datamodule = DataLoader(dataset, collate_fn=ImageBatch.collate, pin_memory=True)
        engine.predict(model=model, dataloaders=datamodule, ckpt_path=str(checkpoint_path))
