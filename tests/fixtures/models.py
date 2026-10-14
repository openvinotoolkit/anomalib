# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared model and checkpoint fixtures for integration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from lightning import seed_everything

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import get_model

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.fixture(scope="session")
def ckpt_path(
    project_path: Path,
    make_dummy_dataset: Callable[[str], Path],
    dataset_path: Path,
) -> Callable[[str], Path]:
    """Return a train-once checkpoint resolver keyed by model name."""
    make_dummy_dataset("mvtecad")
    cache: dict[str, Path] = {}

    def checkpoint(model_name: str) -> Path:
        """Return the cached checkpoint path for a model."""
        if model_name in cache and cache[model_name].exists():
            return cache[model_name]

        seed_everything(0, workers=True)
        model = get_model(model_name)
        checkpoint_root = project_path / "shared_ckpts"
        matches = list(checkpoint_root.glob(f"{model.name}/MVTecAD/dummy/*/weights/lightning/model.ckpt"))
        if not matches:
            engine = Engine(
                logger=False,
                default_root_dir=checkpoint_root,
                max_epochs=1,
                devices=1,
                limit_train_batches=2,
                limit_val_batches=2,
            )
            datamodule = MVTecAD(root=dataset_path / "mvtecad", category="dummy", train_batch_size=2)
            engine.fit(model=model, datamodule=datamodule)
            matches = list(checkpoint_root.glob(f"{model.name}/MVTecAD/dummy/*/weights/lightning/model.ckpt"))
            if not matches:
                msg = f"Checkpoint not found for {model_name}"
                raise FileNotFoundError(msg)

        checkpoint_path = max(matches, key=lambda path: path.stat().st_mtime).resolve()
        cache[model_name] = checkpoint_path
        return checkpoint_path

    return checkpoint
