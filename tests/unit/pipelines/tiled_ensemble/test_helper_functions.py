"""Test ensemble helper functions."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from jsonargparse import Namespace
from lightning.pytorch.callbacks import EarlyStopping

from anomalib.callbacks.normalization import _MinMaxNormalizationCallback
from anomalib.models import AnomalyModule
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
    """Test ensemble helper functions."""

    @staticmethod
    def test_ensemble_datamodule(get_ensemble_config: dict, get_tiler: EnsembleTiler) -> None:
        """Test that datamodule is created and has correct collate function."""
        config = get_ensemble_config
        tiler = get_tiler
        datamodule = get_ensemble_datamodule(config, tiler, (0, 0))

        assert isinstance(datamodule.collate_fn, TileCollater)

    @staticmethod
    def test_ensemble_model(get_ensemble_config: dict, get_tiler: EnsembleTiler) -> None:
        """Test that model is successfully created with correct input shape."""
        config = get_ensemble_config
        tiler = get_tiler
        model = get_ensemble_model(config["TrainModels"]["model"], tiler)

        assert model.input_size == tuple(config["tiling"]["tile_size"])

    @staticmethod
    def test_tiler(get_ensemble_config: dict) -> None:
        """Test that tiler is successfully instantiated."""
        config = get_ensemble_config

        tiler = get_ensemble_tiler(config["tiling"], config["data"])
        assert isinstance(tiler, EnsembleTiler)

    @staticmethod
    def test_trainer_kwargs(get_ensemble_config: dict) -> None:
        """Test that objects are correctly constructed from kwargs."""
        config = get_ensemble_config

        objects = parse_trainer_kwargs(config["TrainModels"]["trainer"])
        assert isinstance(objects, Namespace)
        # verify that early stopping is parsed and added to callbacks
        assert isinstance(objects.callbacks[0], EarlyStopping)

    @staticmethod
    @pytest.mark.parametrize(
        "normalization_stage",
        [NormalizationStage.NONE, NormalizationStage.IMAGE, NormalizationStage.TILE],
    )
    def test_threshold_values(normalization_stage: NormalizationStage, get_mock_stats_dir: Path) -> None:
        """Test that threshold values are correctly set based on normalization stage."""
        stats_dir = get_mock_stats_dir

        i_thresh, p_thresh = get_threshold_values(normalization_stage, stats_dir)

        if normalization_stage != NormalizationStage.NONE:
            # minmax normalization sets thresholds to 0.5
            assert i_thresh == p_thresh == 0.5
        else:
            assert i_thresh == p_thresh == 0.1111


class TestEnsembleEngine:
    """Test ensemble engine configuration."""

    @staticmethod
    @pytest.mark.parametrize(
        "normalization_stage",
        [NormalizationStage.NONE, NormalizationStage.IMAGE, NormalizationStage.TILE],
    )
    def test_normalisation(normalization_stage: NormalizationStage, get_model: AnomalyModule) -> None:
        """Test that normalization callback is correctly initialized."""
        engine = get_ensemble_engine(
            tile_index=(0, 0),
            accelerator="cpu",
            devices="1",
            root_dir=Path("mock"),
            normalization_stage=normalization_stage,
        )

        engine._setup_anomalib_callbacks(get_model)  # noqa: SLF001

        # verify that only in case of tile level normalization the callback is present
        if normalization_stage == NormalizationStage.TILE:
            assert any(
                isinstance(x, _MinMaxNormalizationCallback)
                for x in engine._cache.args["callbacks"]  # noqa: SLF001
            )
        else:
            assert not any(
                isinstance(x, _MinMaxNormalizationCallback)
                for x in engine._cache.args["callbacks"]  # noqa: SLF001
            )
