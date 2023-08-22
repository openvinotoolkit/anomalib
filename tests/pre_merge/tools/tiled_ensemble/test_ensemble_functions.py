"""Test ensemble helper functions"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from tempfile import TemporaryDirectory
from typing import List

from torch import Tensor

from tools.tiled_ensemble.ensemble_functions import (
    TileCollater,
    get_ensemble_callbacks,
    get_ensemble_datamodule,
    get_prediction_storage,
)
from tools.tiled_ensemble.post_processing.postprocess import NormalizationStage
from tools.tiled_ensemble.predictions import (
    MemoryEnsemblePredictions,
    FileSystemEnsemblePredictions,
    DownscaledEnsemblePredictions,
)
from anomalib.utils.callbacks import MinMaxNormalizationCallback
from tools.tiled_ensemble.predictions.prediction_data import PredictionStorageType


class TestTileCollater:
    """Test tile collater"""

    def test_collate_tile_shape(self, get_ensemble_config, get_datamodule):
        config = get_ensemble_config
        # datamodule with tile collater
        datamodule = get_datamodule(config, "segmentation")

        tile_w, tile_h = config.ensemble.tiling.tile_size
        batch_size = config.dataset.train_batch_size

        batch = next(iter(datamodule.train_dataloader()))
        assert batch["image"].shape == (batch_size, 3, tile_w, tile_h)
        assert batch["mask"].shape == (batch_size, tile_w, tile_h)

    def test_collate_box_data(self, get_ensemble_config, get_datamodule):
        config = get_ensemble_config
        # datamodule with tile collater
        datamodule = get_datamodule(config, "detection")

        batch = next(iter(datamodule.train_dataloader()))

        # assert that base collate function was called
        assert isinstance(batch["boxes"], List)
        assert isinstance(batch["boxes"][0], Tensor)


class TestHelperFunctions:
    """Test other ensemble helper functions"""

    def test_ensemble_config(self, get_ensemble_config):
        config = get_ensemble_config

        assert "ensemble" in config
        assert config.model.input_size == config.ensemble.tiling.tile_size

    def test_ensemble_datamodule(self, get_ensemble_config, get_tiler):
        config = get_ensemble_config
        tiler = get_tiler
        datamodule = get_ensemble_datamodule(config, tiler)

        assert isinstance(datamodule.collate_fn, TileCollater)

    def test_ensemble_prediction_storage_type(self, get_ensemble_config):
        config = get_ensemble_config

        config.ensemble.predictions.storage = PredictionStorageType.MEMORY
        pred_direct = get_prediction_storage(config)

        assert isinstance(pred_direct[0], MemoryEnsemblePredictions)
        assert isinstance(pred_direct[1], MemoryEnsemblePredictions)

        config.ensemble.predictions.storage = PredictionStorageType.FILE_SYSTEM

        with TemporaryDirectory() as project_dir:
            config.project.path = project_dir

            pred_fs = get_prediction_storage(config)
            assert isinstance(pred_fs[0], FileSystemEnsemblePredictions)
            assert isinstance(pred_fs[1], FileSystemEnsemblePredictions)

        config.ensemble.predictions.storage = PredictionStorageType.MEMORY_DOWNSCALED
        pred_rescale = get_prediction_storage(config)

        assert isinstance(pred_rescale[0], DownscaledEnsemblePredictions)
        assert isinstance(pred_rescale[1], DownscaledEnsemblePredictions)

    def test_ensemble_prediction_storage_reference(self, get_ensemble_config):
        config = get_ensemble_config

        config.dataset.val_split_mode = "same_as_test"
        pred_same = get_prediction_storage(config)

        # if test data is same as validation, storage should be same
        assert pred_same[0] is pred_same[1]

        config.dataset.val_split_mode = "from_test"
        pred_from = get_prediction_storage(config)

        # if test data is NOT same as validation, storage should be separate object
        assert pred_from[0] is not pred_from[1]

    def test_ensemble_callbacks(self, get_ensemble_config):
        config = get_ensemble_config
        config.ensemble.post_processing.normalization = NormalizationStage.IMAGE

        callbacks = get_ensemble_callbacks(config, (0, 0))

        # if we normalize joined images, minmax callback shouldn't be present here
        for callback in callbacks:
            assert not isinstance(callback, MinMaxNormalizationCallback)
