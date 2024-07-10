"""Test normalization callback."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from lightning.pytorch import seed_everything
from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT

from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import Padim


def run_train_test(normalization_method: str, dataset_path: Path) -> _EVALUATE_OUTPUT:
    """Run training and testing with a given normalization method.

    Args:
        normalization_method (str): Normalization method used to run the test.
        dataset_path (Path): Path to the dummy dataset.

    Returns:
        _EVALUATE_OUTPUT: Results of the test.
    """
    model = Padim()
    datamodule = MVTec(root=dataset_path / "mvtec", category="dummy", seed=42)

    engine = Engine(
        normalization=normalization_method,
        threshold="F1AdaptiveThreshold",
        image_metrics=["F1Score", "AUROC"],
        devices=1,
    )
    engine.fit(model=model, datamodule=datamodule)
    return engine.test(model=model, datamodule=datamodule)


@pytest.mark.skip(reason="This test is flaky and needs to be revisited.")
def test_normalizer(dataset_path: Path) -> None:
    """Test if all normalization methods give the same performance."""
    # run without normalization
    seed_everything(42)
    results_without_normalization = run_train_test(normalization_method="none", dataset_path=dataset_path)

    # run without normalization
    seed_everything(42)
    results_with_minmax_normalization = run_train_test(normalization_method="min_max", dataset_path=dataset_path)

    # performance should be the same
    for metric in ["image_AUROC", "image_F1Score"]:
        assert round(results_without_normalization[0][metric], 3) == round(
            results_with_minmax_normalization[0][metric],
            3,
        )
