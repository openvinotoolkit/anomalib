"""Test ensemble helper functions"""

from tempfile import TemporaryDirectory
from typing import List

import pytest
from torch import Tensor

from anomalib.config import get_configurable_parameters
from anomalib.models.ensemble import EnsembleTiler
from anomalib.models.ensemble.ensemble_functions import (
    TileCollater,
    prepare_ensemble_configurable_parameters,
    get_ensemble_datamodule,
    get_prediction_storage,
    get_ensemble_callbacks,
)
from anomalib.models.ensemble.predictions import (
    BasicEnsemblePredictions,
    FileSystemEnsemblePredictions,
    RescaledEnsemblePredictions,
)
from anomalib.utils.callbacks import MinMaxNormalizationCallback


def get_datamodule(config, task):
    tiler = EnsembleTiler(config)
    config.dataset.task = task

    datamodule = get_ensemble_datamodule(config, tiler)

    datamodule.prepare_data()
    datamodule.setup()

    return datamodule

@pytest.fixture
def get_config():
    config = get_configurable_parameters(config_path="tests/pre_merge/models/ensemble/dummy_padim_config.yaml")
    prepare_ensemble_configurable_parameters(
        ens_config_path="tests/pre_merge/models/ensemble/dummy_ens_config.yaml", config=config
    )
    return config

class TestTileCollater:
    """Test tile collater"""

    def test_collate_tile_shape(self, get_config):
        config = get_config
        # datamodule with tile collater
        datamodule = get_datamodule(config, "segmentation")

        tile_w, tile_h = config.ensemble.tiling.tile_size
        batch_size = config.dataset.train_batch_size

        batch = next(iter(datamodule.train_dataloader()))
        assert batch["image"].shape == (batch_size, 3, tile_w, tile_h)
        assert batch["mask"].shape == (batch_size, tile_w, tile_h)

    def test_collate_box_data(self, get_config):
        config = get_config
        # datamodule with tile collater
        datamodule = get_datamodule(config, "detection")

        batch = next(iter(datamodule.train_dataloader()))

        # assert that base collate function was called
        assert isinstance(batch["boxes"], List)
        assert isinstance(batch["boxes"][0], Tensor)


class TestHelperFunctions:
    """Test other ensemble helper functions"""

    def test_ensemble_config(self, get_config):
        config = get_config

        assert "ensemble" in config
        assert config.model.input_size == config.ensemble.tiling.tile_size

    def test_ensemble_datamodule(self, get_config):
        config = get_config

        tiler = EnsembleTiler(config)
        datamodule = get_ensemble_datamodule(config, tiler)

        assert isinstance(datamodule.custom_collate_fn, TileCollater)

    def test_ensemble_prediction_storage_type(self, get_config):
        config = get_config

        config.ensemble.predictions.storage = "direct"
        pred_direct = get_prediction_storage(config)

        assert isinstance(pred_direct[0], BasicEnsemblePredictions)
        assert isinstance(pred_direct[1], BasicEnsemblePredictions)

        config.ensemble.predictions.storage = "file_system"

        with TemporaryDirectory() as project_dir:
            config.project.path = project_dir

            pred_fs = get_prediction_storage(config)
            assert isinstance(pred_fs[0], FileSystemEnsemblePredictions)
            assert isinstance(pred_fs[1], FileSystemEnsemblePredictions)

        config.ensemble.predictions.storage = "rescaled"
        pred_rescale = get_prediction_storage(config)

        assert isinstance(pred_rescale[0], RescaledEnsemblePredictions)
        assert isinstance(pred_rescale[1], RescaledEnsemblePredictions)

    def test_ensemble_prediction_storage_reference(self, get_config):
        config = get_config

        config.dataset.val_split_mode = "same_as_test"
        pred_same = get_prediction_storage(config)

        # if test data is same as validation, storage should be same
        assert pred_same[0] is pred_same[1]

        config.dataset.val_split_mode = "from_test"
        pred_from = get_prediction_storage(config)

        # if test data is NOT same as validation, storage should be separate object
        assert pred_from[0] is not pred_from[1]

    def test_ensemble_callbacks(self, get_config):
        config = get_config
        config.ensemble.post_processing.normalization = "final"

        callbacks = get_ensemble_callbacks(config, (0, 0))

        for callback in callbacks:
            assert not isinstance(callback, MinMaxNormalizationCallback)
