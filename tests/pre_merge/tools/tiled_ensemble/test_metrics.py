"""Test ensemble metrics"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from anomalib.utils.metrics import AnomalibMetricCollection

mock_predictions = {
    "pred_scores": torch.ones(50),
    "label": torch.ones(50),
    "anomaly_maps": torch.ones(10, 50),
    "mask": torch.ones(10, 50),
}


def test_metrics_setup(get_ens_metrics):
    metrics = get_ens_metrics

    assert isinstance(metrics.image_metrics, AnomalibMetricCollection)
    assert isinstance(metrics.pixel_metrics, AnomalibMetricCollection)


def test_data_unchanged(get_ens_metrics):
    metrics = get_ens_metrics

    metric_out = metrics.process(mock_predictions)

    for name, values in metric_out.items():
        if isinstance(values, Tensor):
            assert values.equal(mock_predictions[name]), f"{name} changed"
        elif isinstance(values, list) and isinstance(values[0], Tensor):
            assert values[0].equal(mock_predictions[name][0]), f"{name} changed"
        else:
            assert values == mock_predictions[name], f"{name} changed"


def test_metric_compute(get_ens_metrics):
    metrics = get_ens_metrics

    metrics.process(mock_predictions)

    compute_out = metrics.compute()

    assert list(compute_out.keys()) == ["image_F1Score", "image_AUROC", "pixel_F1Score", "pixel_AUROC"]
