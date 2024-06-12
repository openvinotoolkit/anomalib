"""Helper functions for the tiled ensemble training."""

from anomalib.data import AnomalibDataModule, get_datamodule
from anomalib.models import AnomalyModule, get_model

from .ensemble_tiling import EnsembleTiler, TileCollater

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


def get_ensemble_datamodule(config: dict, tiler: EnsembleTiler, tile_index: tuple[int, int]) -> AnomalibDataModule:
    """Get Anomaly Datamodule adjusted for use in ensemble.

    Datamodule collate function gets replaced by TileCollater in order to tile all images before they are passed on.

    Args:
        config (dict): Configuration of the anomaly model.
        tiler (EnsembleTiler): Tiler used to split the images to tiles for use in ensemble.
        tile_index (tuple[int, int]): Index of the tile in the split image.

    Returns:
        AnomalibDataModule: Anomalib Lightning DataModule
    """
    datamodule = get_datamodule(config)
    # set custom collate function that does the tiling
    datamodule.collate_fn = TileCollater(tiler, tile_index)
    return datamodule


def get_ensemble_model(config: dict, tiler: EnsembleTiler) -> AnomalyModule:
    """Get model prepared for ensemble training.

    Args:
        config (dict): model configuration.
        tiler (EnsembleTiler): tiler used to get tile dimensions.

    Returns:
        AnomalyModule: model with input_size setup
    """
    model = get_model(config)
    # set model input size match tile size
    model.set_input_size((tiler.tile_size_h, tiler.tile_size_w))

    return model
