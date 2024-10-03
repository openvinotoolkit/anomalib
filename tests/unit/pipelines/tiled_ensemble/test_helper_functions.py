"""Test ensemble helper functions"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from jsonargparse import Namespace
from lightning.pytorch.callbacks import EarlyStopping

from anomalib.callbacks.normalization import _MinMaxNormalizationCallback
from anomalib.pipelines.tiled_ensemble.components.utils import NormalizationStage
from anomalib.pipelines.tiled_ensemble.components.utils.ensemble_tiling import EnsembleTiler, TileCollater
from anomalib.pipelines.tiled_ensemble.components.utils.helper_functions import (
    get_ensemble_datamodule,
    get_ensemble_engine,
    get_ensemble_model,
    get_ensemble_tiler,
    get_threshold_values,
    parse_trainer_kwargs,
)


class TestHelperFunctions:
    """Test ensemble helper functions"""

    def test_ensemble_datamodule(self, get_ensemble_config, get_tiler):
        config = get_ensemble_config
        tiler = get_tiler
        datamodule = get_ensemble_datamodule(config, tiler, (0, 0))

        assert isinstance(datamodule.collate_fn, TileCollater)

    def test_ensemble_model(self, get_ensemble_config, get_tiler):
        config = get_ensemble_config
        tiler = get_tiler
        model = get_ensemble_model(config["TrainModels"]["model"], tiler)

        assert model.input_size == tuple(config["tiling"]["tile_size"])

    def test_tiler(self, get_ensemble_config):
        config = get_ensemble_config

        tiler = get_ensemble_tiler(config["tiling"], config["data"])
        assert isinstance(tiler, EnsembleTiler)

    def test_trainer_kwargs(self, get_ensemble_config):
        config = get_ensemble_config

        objects = parse_trainer_kwargs(config["TrainModels"]["trainer"])
        assert isinstance(objects, Namespace)
        # verify that early stopping is parsed and added to callbacks
        assert isinstance(objects.callbacks[0], EarlyStopping)

    @pytest.fixture(scope="class")
    @staticmethod
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

    @pytest.mark.parametrize(
        "normalization_stage",
        [NormalizationStage.NONE, NormalizationStage.IMAGE, NormalizationStage.TILE],
    )
    def test_threshold_values(self, normalization_stage, get_mock_stats_dir):
        stats_dir = get_mock_stats_dir

        i_thresh, p_thresh = get_threshold_values(normalization_stage, stats_dir)

        if normalization_stage != NormalizationStage.NONE:
            # minmax normalization sets thresholds to 0.5
            assert i_thresh == p_thresh == 0.5
        else:
            assert i_thresh == p_thresh == 0.1111


class TestEnsembleEngine:
    """Test ensemble engine configuration."""

    @pytest.mark.parametrize(
        "normalization_stage",
        [NormalizationStage.NONE, NormalizationStage.IMAGE, NormalizationStage.TILE],
    )
    def test_normalisation(self, normalization_stage):
        engine = get_ensemble_engine(
            tile_index=(0, 0),
            accelerator="cpu",
            devices="1",
            root_dir=Path("mock"),
            normalization_stage=normalization_stage,
        )

        engine._setup_anomalib_callbacks()

        # verify that only in case of tile level normalization the callback is present
        if normalization_stage == NormalizationStage.TILE:
            assert any(map(lambda x: isinstance(x, _MinMaxNormalizationCallback), engine._cache.args["callbacks"]))
        else:
            assert not any(map(lambda x: isinstance(x, _MinMaxNormalizationCallback), engine._cache.args["callbacks"]))
