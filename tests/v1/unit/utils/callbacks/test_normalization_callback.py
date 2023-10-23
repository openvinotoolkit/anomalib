"""Test normalization callback."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from lightning.pytorch import seed_everything

from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import Padim
from tests.helpers.dataset import TestDataset, get_dataset_path


def run_train_test(normalization_method: str, root: str, category: str):
    model = Padim()
    datamodule = MVTec(root=root, category=category)

    engine = Engine(
        normalization=normalization_method,
        threshold="F1AdaptiveThreshold",
        image_metrics=["F1Score", "AUROC"],
        devices=1,
    )
    engine.fit(model=model, datamodule=datamodule)
    results = engine.test(model=model, datamodule=datamodule)
    return results


@TestDataset(num_train=200, num_test=30, path=get_dataset_path(), seed=42)
def test_normalizer(path=get_dataset_path(), category="shapes"):
    # run without normalization
    seed_everything(42)
    results_without_normalization = run_train_test(normalization_method="none", root=path, category=category)

    # run with cdf normalization
    seed_everything(42)
    results_with_cdf_normalization = run_train_test(normalization_method="cdf", root=path, category=category)

    # run without normalization
    seed_everything(42)
    results_with_minmax_normalization = run_train_test(normalization_method="min_max", root=path, category=category)

    # performance should be the same
    for metric in ["image_AUROC", "image_F1Score"]:
        assert round(results_without_normalization[0][metric], 3) == round(results_with_cdf_normalization[0][metric], 3)
        assert round(results_without_normalization[0][metric], 3) == round(
            results_with_minmax_normalization[0][metric], 3
        )
