"""Test SPRO metric."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import pathlib
import tempfile

import torch

from anomalib.metrics.spro import SPRO


def test_spro() -> None:
    """Checks if SPRO metric computes the score utilizing the given saturation configs."""
    saturation_config = [
        {
            "pixel_value": 255,
            "saturation_threshold": 10,
            "relative_saturation": False,
        },
        {
            "pixel_value": 254,
            "saturation_threshold": 0.5,
            "relative_saturation": True,
        },
    ]

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(saturation_config, f)
        saturation_config_json = f.name

    masks = [
        torch.Tensor(
            [
                [
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                ],
                [
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                ],
            ],
        ),
    ]

    masks[0][0] *= 255
    masks[0][1] *= 254

    preds = (torch.arange(8) / 10) + 0.05
    # metrics receive squeezed predictions (N, H, W)
    preds = preds.unsqueeze(1).repeat(1, 5).view(1, 8, 5)

    thresholds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    targets = [1.0, 1.0, 1.0, 0.75, 0.0, 0.0]
    targets_wo_saturation = [1.0, 0.625, 0.5, 0.375, 0.0, 0.0]
    for threshold, target, target_wo_saturation in zip(thresholds, targets, targets_wo_saturation, strict=True):
        # test using saturation_cofig
        spro = SPRO(threshold=threshold, saturation_config=saturation_config_json)
        spro.update(preds, masks)
        assert spro.compute() == target

        # test without saturation_config
        spro_wo_saturaton = SPRO(threshold=threshold)
        spro_wo_saturaton.update(preds, masks)
        assert spro_wo_saturaton.compute() == target_wo_saturation

    # Remove the temporary config file
    pathlib.Path(saturation_config_json).unlink()
