# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test ensemble helper functions."""

from pathlib import Path

import pytest
from jsonargparse import Namespace
from lightning.pytorch.callbacks import EarlyStopping
from torchvision.transforms.v2 import Compose, Normalize, Resize

from anomalib.pipelines.tiled_ensemble.components.utils import NormalizationStage
from anomalib.pipelines.tiled_ensemble.components.utils.ensemble_tiling import EnsembleTiler, TileCollater
from anomalib.pipelines.tiled_ensemble.components.utils.helper_functions import (
    drop_resize_transform,
    get_ensemble_datamodule,
    get_ensemble_model,
    get_ensemble_tiler,
    get_threshold_values,
    parse_trainer_kwargs,
)
from anomalib.pre_processing import PreProcessor
from anomalib.pre_processing.utils.spec import transform_to_spec


class TestHelperFunctions:
    """Test ensemble helper functions."""

    @staticmethod
    def test_ensemble_datamodule_collate_attr(get_ensemble_config: dict, get_tiler: EnsembleTiler) -> None:
        """Test that datamodule is created and has correct collate function."""
        config = get_ensemble_config
        tiler = get_tiler
        datamodule = get_ensemble_datamodule(config, config["tiling"]["image_size"], tiler, (0, 0))

        assert isinstance(datamodule.external_collate_fn, TileCollater)

    @staticmethod
    def test_ensemble_datamodule_img_size(get_ensemble_config: dict, get_tiler: EnsembleTiler) -> None:
        """Test that datamodule is created and has correct collate function."""
        config = get_ensemble_config
        tiler = get_tiler
        datamodule = get_ensemble_datamodule(config, config["tiling"]["image_size"], tiler, (0, 0))

        data = datamodule.test_data[0]

        # check that transforms were correctly set to dataset
        assert list(data.image.shape[1:]) == config["tiling"]["image_size"]

    @staticmethod
    def test_ensemble_model(get_ensemble_config: dict) -> None:
        """Test that model is successfully created with correct input shape."""
        config = get_ensemble_config
        model = get_ensemble_model(
            config["TrainModels"]["model"],
            normalization_stage=config["normalization_stage"],
            input_size=config["tiling"]["tile_size"],
        )
        if hasattr(model.model, "input_size"):
            # if model has fixed input_size, it should be set to tile_size
            assert model.model.input_size == tuple(config["tiling"]["tile_size"])

    @staticmethod
    @pytest.mark.parametrize(
        "normalization_stage",
        [NormalizationStage.NONE, NormalizationStage.IMAGE, NormalizationStage.TILE],
    )
    def test_normalisation_stage(get_ensemble_config: dict, normalization_stage: NormalizationStage) -> None:
        """Test that model postprocessor has normalisation enabled only on tile level."""
        config = get_ensemble_config
        model = get_ensemble_model(
            config["TrainModels"]["model"],
            normalization_stage=normalization_stage,
            input_size=config["tiling"]["tile_size"],
        )

        if normalization_stage == NormalizationStage.TILE:
            assert model.post_processor.enable_normalization
        else:
            assert not model.post_processor.enable_normalization

    @staticmethod
    def test_tiler(get_ensemble_config: dict) -> None:
        """Test that tiler is successfully instantiated."""
        config = get_ensemble_config

        tiler = get_ensemble_tiler(config["tiling"])
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


class TestDropResizeTransform:
    """Test drop_resize_transform, in isolation from any specific model.

    Uses synthetic pre-processors instead of instantiating real models (e.g.
    CFM, AiVad) whose constructors download pretrained weights, so this stays
    fast and network-independent.
    """

    @staticmethod
    def test_bare_resize_collapses_to_none() -> None:
        """A pre-processor whose whole transform is a bare Resize collapses to None.

        Regression test: this previously became an empty list (``[]``), which
        raised ``TypeError`` from ``nn.Module.__setattr__`` immediately (since
        ``pre_processor.transform`` already held a registered ``Resize``
        submodule), and would separately have broken ``transform_to_spec``
        during checkpointing had that assignment succeeded.
        """
        pre_processor = PreProcessor(transform=Resize((256, 256)))

        drop_resize_transform(pre_processor)

        assert pre_processor.transform is None
        assert pre_processor.export_transform is None
        # must not raise, and must produce a checkpoint-safe spec
        assert transform_to_spec(pre_processor.transform) is None

    @staticmethod
    def test_no_transform_stays_none() -> None:
        """A pre-processor with no transform stays None, not an empty list."""
        pre_processor = PreProcessor(transform=None)

        drop_resize_transform(pre_processor)

        assert pre_processor.transform is None
        assert transform_to_spec(pre_processor.transform) is None

    @staticmethod
    def test_resize_dropped_from_compose() -> None:
        """Resize is dropped from a Compose, and the remaining transforms are kept."""
        normalize = Normalize(mean=[0.5], std=[0.5])
        pre_processor = PreProcessor(transform=Compose([Resize((256, 256)), normalize]))

        drop_resize_transform(pre_processor)

        assert isinstance(pre_processor.transform, Compose)
        assert pre_processor.transform.transforms == [normalize]
        # must not raise
        transform_to_spec(pre_processor.transform)

    @staticmethod
    def test_non_resize_transform_is_kept() -> None:
        """A transform that is neither Resize nor Compose is left untouched."""
        normalize = Normalize(mean=[0.5], std=[0.5])
        pre_processor = PreProcessor(transform=normalize)

        drop_resize_transform(pre_processor)

        assert pre_processor.transform is normalize
