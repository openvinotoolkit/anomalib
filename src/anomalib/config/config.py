"""Get configurable parameters."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# TODO: This would require a new design.
# TODO: https://jira.devtools.intel.com/browse/IAAALD-149


import inspect
import logging
from collections.abc import Sequence
from datetime import datetime
from importlib import import_module
from pathlib import Path

from jsonargparse import Namespace
from omegaconf import DictConfig, ListConfig, OmegaConf

from .utils import to_tuple, to_yaml

logger = logging.getLogger(__name__)


def get_default_root_directory(config: DictConfig | ListConfig) -> Path:
    """Sets the default root directory."""
    root_dir = config.results_dir.path if config.results_dir.path else "./results"
    model_name = config.model.class_path.split(".")[-1].lower()
    data_name = config.data.class_path.split(".")[-1].lower()
    category = config.data.init_args.category if "category" in config.data.init_args else ""
    # add datetime folder to the path as well so that runs with same configuration are not overwritten
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if config.results_dir.unique else ""
    # loggers should write to results/model/dataset/category/ folder
    default_root_dir = Path(root_dir, model_name, data_name, category, time_stamp)

    return default_root_dir


def update_config(config: DictConfig | ListConfig | Namespace) -> DictConfig | ListConfig | Namespace:
    """Update config.

    Args:
        config: Configurable parameters.

    Returns:
        DictConfig | ListConfig | Namespace: Updated config.
    """
    show_warnings(config)

    # keep track of the original config file because it will be modified
    config_original: DictConfig | ListConfig | Namespace = (
        config.copy() if isinstance(config, DictConfig | ListConfig) else config.clone()
    )

    config = update_input_size_config(config)

    # Project Configs
    project_path = get_default_root_directory(config)
    logger.info(f"Project path set to {(project_path)}")

    (project_path / "weights").mkdir(parents=True, exist_ok=True)
    (project_path / "images").mkdir(parents=True, exist_ok=True)

    # set visualizer path
    if "visualization" in config and (
        config.visualization.image_save_path == "" or config.visualization.image_save_path is None
    ):
        config.visualization.image_save_path = str(project_path / "images")

    config.trainer.default_root_dir = str(project_path)
    config.results_dir.path = str(project_path)

    config = update_nncf_config(config)

    # write the original config for eventual debug (modified config at the end of the function)
    (project_path / "config_original.yaml").write_text(to_yaml(config_original))

    (project_path / "config.yaml").write_text(to_yaml(config))

    return config


def update_input_size_config(config: DictConfig | ListConfig | Namespace) -> DictConfig | ListConfig | Namespace:
    """Update config with image size as tuple, effective input size and tiling stride.

    Convert integer image size parameters into tuples, calculate the effective input size based on image size
    and crop size, and set tiling stride if undefined.

    Args:
        config (DictConfig | ListConfig | Namespace): Configurable parameters object

    Returns:
        DictConfig | ListConfig: Configurable parameters with updated values
    """
    # Image size: Ensure value is in the form [height, width]
    image_size = config.data.init_args.get("image_size")
    if isinstance(image_size, int):
        config.data.init_args.image_size = (image_size,) * 2
    elif isinstance(image_size, ListConfig | Sequence):
        assert len(image_size) == 2, "image_size must be a single integer or tuple of length 2 for width and height."
    else:
        msg = f"image_size must be either int or ListConfig, got {type(image_size)}"
        raise ValueError(msg)

    # Use input size from data to model input. If model input size is defined, warn and override.
    # If input_size is not part of the model parameters, remove it from the config. This is required due to argument
    # linking from the cli.
    model_module = import_module(".".join(config.model.class_path.split(".")[:-1]))
    model_class = getattr(model_module, config.model.class_path.split(".")[-1])

    # Assign center crop
    center_crop = config.data.init_args.get("center_crop", None)
    if center_crop:
        config.data.init_args.center_crop = to_tuple(center_crop)
    config.data.init_args.image_size = to_tuple(config.data.init_args.image_size)

    if "input_size" in inspect.signature(model_class).parameters:
        # Center crop: Ensure value is in the form [height, width], and update input_size
        if center_crop is not None:
            config.model.init_args.input_size = center_crop
            logger.info(f"Setting model size to crop size {center_crop}")
        else:
            logger.info(
                f" Setting model input size {config.model.init_args.get('input_size', None)} to"
                f" dataset size {config.data.init_args.image_size}.",
            )
            config.model.init_args.input_size = config.data.init_args.image_size
        config.model.init_args.input_size = to_tuple(config.model.init_args.input_size)

    elif "input_size" in config.model.init_args:
        # argument linking adds model input size even if it is not present for that model
        del config.model.init_args.input_size

    if "tiling" in config.keys() and config.tiling.apply:
        if isinstance(config.tiling.tile_size, int):
            config.tiling.tile_size = (config.tiling.tile_size,) * 2
        if config.tiling.stride is None:
            config.tiling.stride = config.tiling.tile_size

    return config


def update_nncf_config(config: DictConfig | ListConfig) -> DictConfig | ListConfig:
    """Set the NNCF input size based on the value of the crop_size parameter in the configurable parameters object.

    Args
        config (DictConfig | ListConfig): Configurable parameters of the current run.

    Returns:
        DictConfig | ListConfig: Updated configurable parameters in DictConfig object.
    """
    image_size = config.data.init_args.image_size
    # If the model has input_size param and in case it takes cropped input
    if "input_size" in config.model.init_args:
        image_size = config.model.init_args.input_size
    sample_size = (image_size, image_size) if isinstance(image_size, int) else image_size
    if "optimization" in config.keys():
        if "nncf" in config.optimization.keys():
            if "input_info" not in config.optimization.nncf.keys():
                config.optimization.nncf["input_info"] = {"sample_size": None}
            config.optimization.nncf.input_info.sample_size = [1, 3, *sample_size]
            if config.optimization.nncf.apply:
                if "update_config" in config.optimization.nncf:
                    return OmegaConf.merge(config, config.optimization.nncf.update_config)
    return config


def update_multi_gpu_training_config(config: DictConfig | ListConfig) -> DictConfig | ListConfig:
    """Updates the config to change learning rate based on number of gpus assigned.

    Current behaviour is to ensure only ddp accelerator is used.

    Args:
        config (DictConfig | ListConfig): Configurable parameters for the current run

    Raises:
        ValueError: If unsupported accelerator is passed

    Returns:
        DictConfig | ListConfig: Updated config
    """
    # validate accelerator
    if config.trainer.accelerator is not None:
        if config.trainer.accelerator.lower() != "ddp":
            if config.trainer.accelerator.lower() in ("dp", "ddp_spawn", "ddp2"):
                logger.warn(
                    f"Using accelerator {config.trainer.accelerator.lower()} is discouraged. "
                    f"Please use one of [null, ddp]. Setting accelerator to ddp",
                )
                config.trainer.accelerator = "ddp"
            else:
                msg = f"Unsupported accelerator found: {config.trainer.accelerator}. Should be one of [null, ddp]"
                raise ValueError(
                    msg,
                )
    # Increase learning rate
    # since pytorch averages the gradient over devices, the idea is to
    # increase the learning rate by the number of devices
    if "lr" in config.model:
        # Number of GPUs can either be passed as gpus: 2 or gpus: [0,1]
        n_gpus: int | list = 1
        if "trainer" in config and "gpus" in config.trainer:
            n_gpus = config.trainer.gpus
        lr_scaler = n_gpus if isinstance(n_gpus, int) else len(n_gpus)
        config.model.lr = config.model.lr * lr_scaler
    return config


def show_warnings(config: DictConfig | ListConfig | Namespace) -> None:
    """Show warnings if any based on the configuration settings

    Args:
        config (DictConfig | ListConfig | Namespace): Configurable parameters for the current run.
    """

    if "clip_length_in_frames" in config.data.keys() and config.data.init_args.clip_length_in_frames > 1:
        logger.warn(
            "Anomalib's models and visualizer are currently not compatible with video datasets with a clip length > 1. "
            "Custom changes to these modules will be needed to prevent errors and/or unpredictable behaviour.",
        )


def get_configurable_parameters(
    model_name: str | None = None,
    config_path: Path | str | None = None,
    config_filename: str | None = "config",
    config_file_extension: str | None = "yaml",
) -> DictConfig | ListConfig:
    """Get configurable parameters.

    Args:
        model_name: str | None:  (Default value = None)
        config_path: Path | str | None:  (Default value = None)
        config_filename: str | None:  (Default value = "config")
        config_file_extension: str | None:  (Default value = "yaml")

    Returns:
        DictConfig | ListConfig: Configurable parameters in DictConfig object.
    """
    if model_name is None and config_path is None:
        msg = (
            "Both model_name and model config path cannot be None! "
            "Please provide a model name or path to a config file!"
        )
        raise ValueError(
            msg,
        )

    if model_name == "efficientad":
        msg = "`efficientad` is deprecated as --model. Please use `efficient_ad` instead."
        logger.warn(msg)
        model_name = "efficient_ad"

    if config_path is None:
        config_path = Path(f"src/anomalib/models/{model_name}/{config_filename}.{config_file_extension}")

    config = OmegaConf.load(config_path)

    config = update_config(config)

    return config
