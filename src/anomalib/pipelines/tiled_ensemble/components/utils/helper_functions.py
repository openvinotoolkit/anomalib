# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for the tiled ensemble training."""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from jsonargparse import ArgumentParser, Namespace
from lightning import Trainer
from torchvision.transforms.v2 import Compose, Resize, Transform

from anomalib.data import AnomalibDataModule, ImageBatch, get_datamodule
from anomalib.models import AnomalibModule, get_model
from anomalib.pre_processing.utils.transform import get_exportable_transform

if TYPE_CHECKING:
    from anomalib.post_processing import PostProcessor
    from anomalib.pre_processing import PreProcessor

from . import NormalizationStage
from .ensemble_engine import TiledEnsembleEngine
from .ensemble_tiling import EnsembleTiler, TileCollater

logger = logging.getLogger(__name__)


def get_ensemble_datamodule(
    data_config: dict,
    image_size: int | tuple[int, int],
    tiler: EnsembleTiler,
    tile_index: tuple[int, int],
) -> AnomalibDataModule:
    """Get Anomaly Datamodule adjusted for use in ensemble.

    Datamodule collate function gets replaced by TileCollater in order to tile all images before they are passed on.

    Args:
        data_config: tiled ensemble data configuration.
        image_size (int | tuple[int, int]): full effective image size of tiled ensemble.
        tiler (EnsembleTiler): Tiler used to split the images to tiles for use in ensemble.
        tile_index (tuple[int, int]): Index of the tile in the split image.

    Returns:
        AnomalibDataModule: Anomalib Lightning DataModule
    """
    datamodule = get_datamodule(data_config)
    datamodule.setup()

    # add tiled ensemble image_size transform to datamodule
    setup_transforms(datamodule, image_size=image_size)
    datamodule.external_collate_fn = TileCollater(tiler, tile_index, default_collate_fn=ImageBatch.collate)
    # manually set setup, so later setup doesn't override the transforms...
    datamodule._is_setup = True  # noqa: SLF001

    return datamodule


def setup_transforms(datamodule: AnomalibDataModule, image_size: int | tuple[int, int]) -> None:
    """Modify datamodule resize transforms so the effective ensemble image_size is correct.

    Args:
        datamodule: datamodule where resize transform will be setup.
        image_size (int | tuple[int, int]): tiled ensemble input image size

    """
    resize_transform = Resize(image_size)

    for subset_name in ["train", "val", "test"]:
        default_aug = getattr(datamodule, f"{subset_name}_augmentations", None)

        if isinstance(default_aug, Resize):
            msg = f"Conflicting resize shapes found between dataset augmentations and tiled ensemble size. \
                You are using a Resize transform in your input data augmentations. Please be aware that the \
                tiled ensemble image size is determined by tiling config. The final effective input size as \
                seen by individual model will be determined by the tile_size. To change \
                the effective ensemble input size, please change the image_size in the tiling config. \
                Augmentations: {default_aug.size}, Tiled ensemble base size: {image_size}"
            logger.warning(msg)
            augmentations = resize_transform
        elif isinstance(default_aug, Compose):
            augmentations = Compose([*default_aug.transforms, resize_transform])
        elif isinstance(default_aug, Transform):
            augmentations = Compose([default_aug, resize_transform])
        else:
            augmentations = resize_transform
        # add augmentations with resize to datamodule and datasets, ensuring that output images match effective size
        setattr(datamodule, f"{subset_name}_augmentations", augmentations)
        data_subset = getattr(datamodule, f"{subset_name}_data", None)
        if data_subset is not None:
            data_subset.augmentations = augmentations


def drop_resize_transform(pre_processor: "PreProcessor") -> None:
    """Remove the ``Resize`` transform from a pre-processor's transform pipeline, in place.

    Resize is dropped since it gets copied to the datamodule instead (the
    tiled ensemble determines the effective input size via tiling config, not
    the model's own preprocessor).

    Uses ``None`` rather than an empty list to represent "no transform",
    since ``PreProcessor.transform`` is typed ``Transform | None``, and an
    empty list is neither a registered ``Transform`` (breaking
    ``transform_to_spec`` during checkpointing) nor assignable over a
    previously-registered ``nn.Module`` transform (``nn.Module.__setattr__``
    only accepts ``None`` or another module in that slot).

    Args:
        pre_processor (PreProcessor): Pre-processor to update in place.
    """
    pre_transforms = pre_processor.transform
    if isinstance(pre_transforms, Resize):
        update_transform = None
    elif isinstance(pre_transforms, Compose):
        update_transform = Compose([
            transform for transform in pre_transforms.transforms if not isinstance(transform, Resize)
        ])
    elif pre_transforms is not None:
        update_transform = pre_transforms
    else:
        update_transform = None

    pre_processor.transform = update_transform
    pre_processor.export_transform = get_exportable_transform(update_transform)


def get_ensemble_model(
    model_args: dict,
    input_size: int | tuple[int, int],
    normalization_stage: NormalizationStage,
) -> AnomalibModule:
    """Get model prepared for ensemble training.

    Args:
        model_args (dict): tiled ensemble model configuration.
        input_size (int | tuple[int, int]): individual model input size.
        normalization_stage (NormalizationStage): stage when normalization performed.

    Returns:
        AnomalibModule: model with input_size setup
    """
    # first make temporary model to get object
    temp_model = get_model(model_args)
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    # create custom pre_proc with correct input size
    # since we can't modify input_size directly (needed during instantiation by some models like FastFlow)
    pre_processor = temp_model.configure_pre_processor(input_size)
    # make actual model with correct input size
    model: AnomalibModule = get_model(model_args, pre_processor=pre_processor, visualizer=False)
    if model.pre_processor is not None:
        drop_resize_transform(model.pre_processor)

    if model.post_processor is not None:
        model_post_processor: PostProcessor = model.post_processor
        # set model normalisation only if the stage is set to tile level (but thresholding is always applied)
        model_post_processor.enable_normalization = normalization_stage == NormalizationStage.TILE

    return model


def get_ensemble_tiler(tiling_args: dict) -> EnsembleTiler:
    """Get tiler used for image tiling and to obtain tile dimensions.

    Args:
        tiling_args: tiled ensemble tiling configuration.

    Returns:
        EnsembleTiler: tiler object.
    """
    tiler = EnsembleTiler(
        tile_size=tiling_args["tile_size"],
        stride=tiling_args["stride"],
        image_size=tiling_args["image_size"],
    )

    return tiler  # noqa: RET504


def parse_trainer_kwargs(trainer_args: dict | None) -> Namespace | dict:
    """Parse trainer args and instantiate all needed elements.

    Transforms config into kwargs ready for Trainer, including instantiation of callback etc.

    Args:
        trainer_args (dict): Trainer args dictionary.

    Returns:
        dict: parsed kwargs with instantiated elements.
    """
    if not trainer_args:
        return {}

    # try to get trainer args, if not present return empty
    parser = ArgumentParser()

    parser.add_class_arguments(Trainer, fail_untyped=False, instantiate=False, sub_configs=True)
    config = parser.parse_object(trainer_args)
    objects = parser.instantiate_classes(config)

    return objects  # noqa: RET504


def get_ensemble_engine(
    tile_index: tuple[int, int],
    accelerator: str,
    devices: list[int] | str | int,
    root_dir: Path,
    trainer_args: dict | None = None,
) -> TiledEnsembleEngine:
    """Prepare engine for ensemble training or prediction.

    This method makes sure correct normalization is used, prepares metrics and additional trainer kwargs..

    Args:
        tile_index (tuple[int, int]): Index of tile that this model processes.
        accelerator (str): Accelerator (device) to use.
        devices (list[int] | str | int): device IDs used for training.
        root_dir (Path): Root directory to save checkpoints, stats and images.
        trainer_args (dict): Trainer args dictionary. Empty dict if not present.

    Returns:
        TiledEnsembleEngine: set up engine for ensemble training/prediction.
    """
    # parse additional trainer args and callbacks if present in config
    trainer_kwargs = parse_trainer_kwargs(trainer_args)
    # remove keys that we already have
    trainer_kwargs.pop("accelerator", None)
    trainer_kwargs.pop("default_root_dir", None)
    trainer_kwargs.pop("devices", None)

    # create engine for specific tile location
    engine = TiledEnsembleEngine(
        tile_index=tile_index,
        accelerator=accelerator,
        devices=devices,
        default_root_dir=root_dir,
        **trainer_kwargs,
    )

    return engine  # noqa: RET504


def get_threshold_values(normalization_stage: NormalizationStage, root_dir: Path) -> tuple[float, float]:
    """Get threshold values for image and pixel level predictions.

    If normalization is not used, get values based on statistics obtained from validation set.
    If normalization is used, both image and pixel threshold are 0.5

    Args:
        normalization_stage (NormalizationStage): ensemble run args, used to get normalization stage.
        root_dir (Path): path to run root where stats file is saved.

    Returns:
        tuple[float, float]: image and pixel threshold.
    """
    if normalization_stage == NormalizationStage.NONE:
        stats_path = root_dir / "weights" / "lightning" / "stats.json"
        with stats_path.open("r") as f:
            stats = json.load(f)
        image_threshold = stats["image_threshold"]
        pixel_threshold = stats["pixel_threshold"]
    else:
        # normalization transforms the scores so that threshold is at 0.5
        image_threshold = 0.5
        pixel_threshold = 0.5

    return image_threshold, pixel_threshold
