"""Helper functions for the tiled ensemble training."""
from pathlib import Path

from anomalib.data import AnomalibDataModule, get_datamodule
from anomalib.models import AnomalyModule, get_model
from anomalib.utils.normalization import NormalizationMethod

from .ensemble_engine import TiledEnsembleEngine
from .ensemble_tiling import EnsembleTiler, TileCollater
from .post_processing.postprocess import NormalizationStage

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


def get_ensemble_datamodule(data_config: dict, tiler: EnsembleTiler, tile_index: tuple[int, int]) -> AnomalibDataModule:
    """Get Anomaly Datamodule adjusted for use in ensemble.

    Datamodule collate function gets replaced by TileCollater in order to tile all images before they are passed on.

    Args:
        data_config (dict): Configuration of the anomaly model.
        tiler (EnsembleTiler): Tiler used to split the images to tiles for use in ensemble.
        tile_index (tuple[int, int]): Index of the tile in the split image.

    Returns:
        AnomalibDataModule: Anomalib Lightning DataModule
    """
    datamodule = get_datamodule(data_config)
    # set custom collate function that does the tiling
    datamodule.collate_fn = TileCollater(tiler, tile_index)
    datamodule.setup()

    return datamodule


def get_ensemble_model(model_config: dict, tiler: EnsembleTiler) -> AnomalyModule:
    """Get model prepared for ensemble training.

    Args:
        model_config (dict): model configuration.
        tiler (EnsembleTiler): tiler used to get tile dimensions.

    Returns:
        AnomalyModule: model with input_size setup
    """
    model = get_model(model_config)
    # set model input size match tile size
    model.set_input_size((tiler.tile_size_h, tiler.tile_size_w))

    return model


def get_ensemble_tiler(args: dict) -> EnsembleTiler:
    """Get tiler used for image tiling and to obtain tile dimensions.

    Args:
        args: tiled ensemble run configuration.

    Returns:
        EnsembleTiler: tiler object.
    """
    tiler = EnsembleTiler(
        tile_size=args["ensemble"]["tiling"]["tile_size"],
        stride=args["ensemble"]["tiling"]["stride"],
        image_size=args["data"]["init_args"]["image_size"],
    )

    return tiler  # noqa: RET504


def get_ensemble_engine(
    tile_index: tuple[int, int],
    accelerator: str,
    devices: list[int] | str | int,
    root_dir: Path,
    post_process_config: dict,
) -> TiledEnsembleEngine:
    """Prepare engine for ensemble training or prediction.

    This method makes sure correct normalization is used and sets the root_dir.

    Args:
        tile_index (tuple[int, int]): Index of tile that this model processes.
        accelerator (str): Accelerator (device) to use.
        devices (list[int] | str | int): device IDs used for training.
        root_dir (Path): Root directory to save checkpoints, stats and images.
        post_process_config (dict): Config dictionary for ensemble post-processing.

    Returns:
        TiledEnsembleEngine: set up engine for ensemble training/prediction.
    """
    # if we want tile level normalization we set it here, otherwise it's done later on joined images
    if post_process_config["normalization_stage"] == NormalizationStage.TILE:
        normalization = NormalizationMethod.MIN_MAX
    else:
        normalization = NormalizationMethod.NONE

    # create engine for specific tile location and fit the model
    engine = TiledEnsembleEngine(
        tile_index=tile_index,
        normalization=normalization,
        accelerator=accelerator,
        devices=devices,
        default_root_dir=root_dir,
    )

    return engine  # noqa: RET504
