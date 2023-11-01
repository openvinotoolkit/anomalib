"""Test normalization callback."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from lightning.pytorch import seed_everything
from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT

from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import Padim
from tests.legacy.helpers.dataset import get_dataset_path


def run_train_test(normalization_method: str) -> _EVALUATE_OUTPUT:
    """Run training and testing with a given normalization method.

    Args:
        normalization_method (str): Normalization method used to run the test.

    Returns:
        _EVALUATE_OUTPUT: Results of the test.
    """
    model = Padim()
    datamodule = MVTec(root=get_dataset_path(), category="bottle", seed=42)

    engine = Engine(
        normalization=normalization_method,
        threshold="F1AdaptiveThreshold",
        image_metrics=["F1Score", "AUROC"],
        devices=1,
    )
    engine.fit(model=model, datamodule=datamodule)
    return engine.test(model=model, datamodule=datamodule)


def test_normalizer() -> None:
    """Test if all normalization methods give the same performance."""
    # run without normalization
    seed_everything(42)
    results_without_normalization = run_train_test(normalization_method="none")

    # run with cdf normalization
    seed_everything(42)
    results_with_cdf_normalization = run_train_test(normalization_method="cdf")

    # run without normalization
    seed_everything(42)
    results_with_minmax_normalization = run_train_test(normalization_method="min_max")

    # performance should be the same
    for metric in ["image_AUROC", "image_F1Score"]:
        assert round(results_without_normalization[0][metric], 3) == round(results_with_cdf_normalization[0][metric], 3)
        assert round(results_without_normalization[0][metric], 3) == round(
            results_with_minmax_normalization[0][metric],
            3,
        )
