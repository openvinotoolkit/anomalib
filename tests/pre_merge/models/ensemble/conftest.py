"""Fixtures that are used in ensemble testing"""

import pytest

from anomalib.config import get_configurable_parameters
from anomalib.models.ensemble import EnsembleTiler, get_ensemble_datamodule, prepare_ensemble_configurable_parameters
from anomalib.models.ensemble.predictions import BasicEnsemblePredictions, BasicPredictionJoiner


@pytest.fixture(scope="module")
def get_config():
    config = get_configurable_parameters(config_path="tests/pre_merge/models/ensemble/dummy_padim_config.yaml")
    prepare_ensemble_configurable_parameters(
        ens_config_path="tests/pre_merge/models/ensemble/dummy_ens_config.yaml", config=config
    )
    return config


@pytest.fixture(scope="module")
def get_tiler(get_config):
    return EnsembleTiler(get_config)


@pytest.fixture(scope="module")
def get_datamodule(get_config, get_tiler):
    tiler = get_tiler

    def get_datamodule(config, task):
        config.dataset.task = task

        datamodule = get_ensemble_datamodule(config, tiler)

        datamodule.prepare_data()
        datamodule.setup()

        return datamodule

    return get_datamodule


@pytest.fixture(scope="module")
def get_joiner(get_config, get_tiler):
    tiler = get_tiler

    joiner = BasicPredictionJoiner(tiler)

    return joiner


@pytest.fixture(scope="module")
def get_ensemble_predictions(get_datamodule, get_config):
    config = get_config
    datamodule = get_datamodule(config, "segmentation")

    data = BasicEnsemblePredictions()

    for tile_index in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        datamodule.setup()
        datamodule.custom_collate_fn.tile_index = tile_index

        tile_prediction = []
        batch = next(iter(datamodule.test_dataloader()))
        # set mock maps to just one channel of image
        batch["anomaly_maps"] = batch["image"].clone()[:, 0, :, :].unsqueeze(1)
        # set mock pred mask to mask but add channel
        batch["pred_masks"] = batch["mask"].clone().unsqueeze(1)
        tile_prediction.append(batch)

        # store to prediction storage object
        data.add_tile_prediction(tile_index, tile_prediction)

    return data
