# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fixtures that are used in tiled ensemble testing."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
import yaml

from anomalib.data import AnomalibDataModule, ImageBatch
from anomalib.models import AnomalibModule
from anomalib.pipelines.tiled_ensemble.components.utils.ensemble_tiling import EnsembleTiler
from anomalib.pipelines.tiled_ensemble.components.utils.helper_functions import (
    get_ensemble_datamodule,
    get_ensemble_model,
    get_ensemble_tiler,
)
from anomalib.pipelines.tiled_ensemble.components.utils.prediction_data import EnsemblePredictions
from anomalib.pipelines.tiled_ensemble.components.utils.prediction_merging import PredictionMergingMechanism


@pytest.fixture(scope="module")
def get_ensemble_config(mvtecad_path: Path) -> dict:
    """Return ensemble dummy config dict with corrected dataset path to dummy temp dir."""
    with Path("tests/unit/pipelines/tiled_ensemble/dummy_config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
        # dummy dataset
        config["data"]["init_args"]["root"] = mvtecad_path

        return config


@pytest.fixture(scope="module")
def get_tiler(get_ensemble_config: dict) -> EnsembleTiler:
    """Return EnsembleTiler object based on test dummy config."""
    config = get_ensemble_config

    return get_ensemble_tiler(config["tiling"])


@pytest.fixture(scope="module")
def get_model(get_ensemble_config: dict) -> AnomalibModule:
    """Return model prepared for tiled ensemble training."""
    config = get_ensemble_config

    return get_ensemble_model(
        config["TrainModels"]["model"],
        normalization_stage=config["normalization_stage"],
        input_size=config["tiling"]["tile_size"],
    )


@pytest.fixture(scope="module")
def get_datamodule(get_ensemble_config: dict, get_tiler: EnsembleTiler) -> AnomalibDataModule:
    """Return ensemble datamodule."""
    config = get_ensemble_config
    tiler = get_tiler
    datamodule = get_ensemble_datamodule(config, config["tiling"]["image_size"], tiler, (0, 0))
    datamodule.setup()

    return datamodule


@pytest.fixture(scope="module")
def get_tile_predictions(get_datamodule: AnomalibDataModule) -> EnsemblePredictions:
    """Return tile predictions inside EnsemblePredictions object."""
    datamodule = get_datamodule

    data = EnsemblePredictions()

    for tile_index in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        datamodule.external_collate_fn.tile_index = tile_index

        tile_prediction = []
        batch = next(iter(datamodule.test_dataloader()))

        # make mock labels and scores
        batch.pred_score = torch.rand(batch.gt_label.shape)
        batch.pred_label = batch.pred_score > 0.5

        # set mock maps to just one channel of image
        batch.anomaly_map = batch.image.clone()[:, 0, :, :].unsqueeze(1)
        # set mock pred mask to mask but add channel
        batch.pred_mask = batch.gt_mask.clone().unsqueeze(1)

        tile_prediction.append(batch)

        # store to prediction storage object
        data.add_tile_prediction(tile_index, tile_prediction)

    return data


@pytest.fixture(scope="module")
def get_batch_predictions() -> list[dict]:
    """Return mock batched predictions."""
    mock_data = {
        "image": torch.rand((5, 3, 100, 100)),
        "gt_mask": (torch.rand((5, 100, 100)) > 0.5).type(torch.float32),
        "anomaly_map": torch.rand((5, 1, 100, 100)),
        "gt_label": torch.tensor([0, 1, 1, 0, 1]).type(torch.int32),
        "pred_score": torch.rand(5),
        "pred_label": torch.ones(5),
        "pred_mask": torch.zeros((5, 100, 100)),
    }

    return [ImageBatch(**mock_data), ImageBatch(**mock_data)]


@pytest.fixture(scope="module")
def get_merging_mechanism(
    get_tile_predictions: EnsemblePredictions,
    get_tiler: EnsembleTiler,
) -> PredictionMergingMechanism:
    """Return ensemble prediction merging mechanism object."""
    tiler = get_tiler
    predictions = get_tile_predictions
    return PredictionMergingMechanism(predictions, tiler)


@pytest.fixture(scope="module")
def get_mock_stats_dir() -> Path:
    """Get temp dir containing statistics."""
    with TemporaryDirectory() as temp_dir:
        stats = {
            "minmax": {
                "anomaly_maps": {
                    "min": 1.9403648376464844,
                    "max": 209.91940307617188,
                },
                "box_scores": {
                    "min": 0.5,
                    "max": 0.45,
                },
                "pred_scores": {
                    "min": 9.390382766723633,
                    "max": 209.91940307617188,
                },
            },
            "image_threshold": 0.1111,
            "pixel_threshold": 0.1111,
        }
        stats_path = Path(temp_dir) / "weights" / "lightning" / "stats.json"
        stats_path.parent.mkdir(parents=True)

        # save mock statistics
        with stats_path.open("w", encoding="utf-8") as stats_file:
            json.dump(stats, stats_file, ensure_ascii=False, indent=4)

        yield Path(temp_dir)
