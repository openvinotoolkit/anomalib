"""Fixtures that are used in tiled ensemble testing"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from anomalib.config import get_configurable_parameters
from tools.tiled_ensemble import EnsembleTiler, get_ensemble_datamodule, prepare_ensemble_configurable_parameters
from tools.tiled_ensemble.post_processing import EnsembleMetrics
from tools.tiled_ensemble.predictions import MemoryEnsemblePredictions, EnsemblePredictionJoiner


@pytest.fixture(scope="module")
def get_ensemble_config():
    config = get_configurable_parameters(config_path="tests/pre_merge/tools/tiled_ensemble/dummy_padim_config.yaml")
    prepare_ensemble_configurable_parameters(
        ens_config_path="tests/pre_merge/tools/tiled_ensemble/dummy_ens_config.yaml", config=config
    )
    return config


@pytest.fixture(scope="module")
def get_tiler(get_ensemble_config):
    config = get_ensemble_config
    return EnsembleTiler(
        tile_size=config.ensemble.tiling.tile_size,
        stride=config.ensemble.tiling.stride,
        image_size=config.dataset.image_size,
    )


@pytest.fixture(scope="module")
def get_datamodule(get_ensemble_config, get_tiler):
    tiler = get_tiler

    def get_datamodule(config, task):
        config.dataset.task = task

        datamodule = get_ensemble_datamodule(config, tiler)

        datamodule.prepare_data()
        datamodule.setup()

        return datamodule

    return get_datamodule


@pytest.fixture(scope="module")
def get_joiner(get_ensemble_config, get_tiler):
    tiler = get_tiler

    joiner = EnsemblePredictionJoiner(tiler)

    return joiner


@pytest.fixture(scope="module")
def get_ensemble_predictions(get_datamodule, get_ensemble_config):
    config = get_ensemble_config
    datamodule = get_datamodule(config, "segmentation")

    data = MemoryEnsemblePredictions()

    for tile_index in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        datamodule.setup()
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
def get_ensemble_metrics(get_ensemble_config):
    config = get_ensemble_config
    metrics = EnsembleMetrics(
        config.dataset.task,
        config.ensemble.metrics.get("image", None),
        config.ensemble.metrics.get("pixel", None),
        0.5,
        0.5,
    )

    return metrics
