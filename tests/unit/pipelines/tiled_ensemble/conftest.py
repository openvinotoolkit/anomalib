"""Fixtures that are used in tiled ensemble testing"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
import yaml

from anomalib.pipelines.tiled_ensemble.components.utils.helper_functions import (
    get_ensemble_datamodule,
    get_ensemble_model,
    get_ensemble_tiler,
)
from anomalib.pipelines.tiled_ensemble.components.utils.prediction_data import EnsemblePredictions
from anomalib.pipelines.tiled_ensemble.components.utils.prediction_merging import PredictionMergingMechanism


@pytest.fixture(scope="module")
def get_ensemble_config():
    with Path("tests/unit/pipelines/tiled_ensemble/dummy_config.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)


@pytest.fixture(scope="module")
def get_tiler(get_ensemble_config):
    config = get_ensemble_config

    return get_ensemble_tiler(config["tiling"], config["data"])


@pytest.fixture(scope="module")
def get_model(get_ensemble_config, get_tiler):
    config = get_ensemble_config
    tiler = get_tiler

    return get_ensemble_model(config["TrainModels"]["model"], tiler)


@pytest.fixture(scope="module")
def get_datamodule(get_ensemble_config, get_tiler):
    config = get_ensemble_config
    tiler = get_tiler
    datamodule = get_ensemble_datamodule(config, tiler, (0, 0))
    datamodule.setup()

    return datamodule


@pytest.fixture(scope="module")
def get_tile_predictions(get_datamodule, get_ensemble_config):
    datamodule = get_datamodule

    data = EnsemblePredictions()

    for tile_index in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        datamodule.collate_fn.tile_index = tile_index

        tile_prediction = []
        batch = next(iter(datamodule.test_dataloader()))
        batch_size = batch["image"].shape[0]

        # make mock labels and scores
        batch["pred_scores"] = torch.rand(batch["label"].shape)
        batch["pred_labels"] = batch["pred_scores"] > 0.5

        # set mock maps to just one channel of image
        batch["anomaly_maps"] = batch["image"].clone()[:, 0, :, :].unsqueeze(1)
        # set mock pred mask to mask but add channel
        batch["pred_masks"] = batch["mask"].clone().unsqueeze(1)

        # make mock boxes
        batch["pred_boxes"] = [torch.rand(1, 4) for _ in range(batch_size)]
        batch["box_scores"] = [torch.rand(1) for _ in range(batch_size)]
        batch["box_labels"] = [bs > 0.5 for bs in batch["box_scores"]]
        tile_prediction.append(batch)

        # store to prediction storage object
        data.add_tile_prediction(tile_index, tile_prediction)

    return data


@pytest.fixture(scope="module")
def get_batch_predictions():
    mock_data = {
        "image": torch.rand((5, 3, 100, 100)),
        "mask": (torch.rand((5, 100, 100)) > 0.5).type(torch.float32),
        "anomaly_maps": torch.rand((5, 1, 100, 100)),
        "label": torch.Tensor([0, 1, 1, 0, 1]),
        "pred_scores": torch.ones(5),
        "pred_labels": torch.ones(5),
        "pred_masks": torch.zeros((5, 100, 100)),
        "pred_boxes": [torch.rand(1, 4) for _ in range(5)],
        "box_labels": [torch.tensor([0.5]) for _ in range(5)],
        "box_scores": [torch.tensor([0.5]) for _ in range(5)],
    }

    return [mock_data, mock_data]


@pytest.fixture(scope="module")
def get_merging_mechanism(get_tile_predictions, get_tiler):
    tiler = get_tiler
    predictions = get_tile_predictions

    joiner = PredictionMergingMechanism(predictions, tiler)

    return joiner


@pytest.fixture(scope="module")
def get_mock_stats_dir(get_ensemble_config):
    with TemporaryDirectory() as temp_dir:
        stats = {
            "minmax": {"min": 0, "max": 1},
            "image_threshold": 0.1111,
            "pixel_threshold": 0.1111,
        }
        stats_path = Path(temp_dir) / "weights" / "lightning" / "stats.json"
        stats_path.parent.mkdir(parents=True)

        # save mock statistics
        with stats_path.open("w", encoding="utf-8") as stats_file:
            json.dump(stats, stats_file, ensure_ascii=False, indent=4)

        yield Path(temp_dir)
