"""
Configurable Getter
"""

from pathlib import Path
from typing import List, Optional, Union
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


def update_config_for_nncf(config: Union[DictConfig, ListConfig]):
    """
    Set the NNCF input size based on the value of the crop_size parameter in the configurable parameters object.

    Args:
        config: Dictconfig: Configurable parameters of the current run.

    Returns:
        Updated configurable parameters in DictConfig object.
    """
    crop_size = config.transform.crop_size if config.transform.crop_size is not None else config.transform.image_size
    sample_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
    if "optimization" in config.keys():
        if "nncf" in config.optimization.keys():
            config.optimization.nncf.input_info.sample_size = [1, 3, *sample_size]
            if config.optimization.nncf.apply:
                if "update_config" in config.optimization.nncf:
                    return OmegaConf.merge(config, config.optimization.nncf.update_config)
    return config


def update_config_for_multi_gpu_training(config: Union[DictConfig, ListConfig]) -> Union[DictConfig, ListConfig]:
    """Updates the config to change learning rate based on number of gpus assigned
        and ensures only ddp accelerator is used

    Args:
        config (Union[DictConfig, ListConfig]): Configurable parameters for the current run

    Raises:
        ValueError: If unsupported accelerator is passed

    Returns:
        Union[DictConfig, ListConfig]: Updated config
    """
    # validate accelerator
    if config.trainer.accelerator is not None:
        if config.trainer.accelerator.lower() != "ddp":
            if config.trainer.accelerator.lower() in ("dp", "ddp_spawn", "ddp2"):
                warn(
                    f"Using accelerator {config.trainer.accelerator.lower()} is discouraged. "
                    f"Please use one of [null, ddp]. Setting accelerator to ddp"
                )
                config.trainer.accelerator = "ddp"
            else:
                raise ValueError(
                    f"Unsupported accelerator found: {config.trainer.accelerator}. Should be one of [null, ddp]"
                )
    # Increase learning rate
    # since pytorch averages the gradient over devices, the idea is to
    # increase the learning rate by the number of devices
    if "lr" in config.model:
        # Number of GPUs can either be passed as gpus: 2 or gpus: [0,1]
        n_gpus: Union[int, List] = 1
        if "trainer" in config and "gpus" in config.trainer:
            n_gpus = config.trainer.gpus
        lr_scaler = n_gpus if isinstance(n_gpus, int) else len(n_gpus)
        config.model.lr = config.model.lr * lr_scaler
    return config


def get_configurable_parameters(
    model_name: Optional[str] = None,
    model_config_path: Optional[Union[Path, str]] = None,
    weight_file: Optional[str] = None,
    openvino: bool = False,
    config_filename: Optional[str] = "config",
    config_file_extension: Optional[str] = "yaml",
) -> Union[DictConfig, ListConfig]:
    """
    Get configurable parameters

    Args:
        model_name: Optional[str]:  (Default value = None)
        model_config_path: Optional[Union[Path, str]]:  (Default value = None)
        weight_file: Path to the weight file
        openvino: Use OpenVINO
        config_filename: Optional[str]:  (Default value = "config")
        config_file_extension: Optional[str]:  (Default value = "yaml")

    Returns:
        Configurable parameters in DictConfig object.

    """
    if model_name is None and model_config_path is None:
        raise ValueError(
            "Both model_name and model config path cannot be None! "
            "Please provide a model name or path to a config file!"
        )

    if model_config_path is None:
        model_config_path = Path(f"anomalib/models/{model_name}/{config_filename}.{config_file_extension}")

    config = OmegaConf.load(model_config_path)

    # Dataset Configs
    if "format" not in config.dataset.keys():
        config.dataset.format = "mvtec"

    config = update_input_size(config)

    # Project Configs
    project_path = Path(config.project.path) / config.model.name / config.dataset.name / config.dataset.category
    (project_path / "weights").mkdir(parents=True, exist_ok=True)
    (project_path / "images").mkdir(parents=True, exist_ok=True)
    config.project.path = str(project_path)
    # loggers should write to results/model/dataset/category/ folder
    config.trainer.default_root_dir = str(project_path)

    if weight_file:
        config.weight_file = weight_file

    # NNCF Parameters
    config = update_config_for_nncf(config)

    config.openvino = openvino
    if openvino:
        config.trainer.gpus = 0

    # XPU Configuration
    if not torch.cuda.is_available():
        config.trainer.gpus = 0

    # Handle multi-gpu training
    config = update_config_for_multi_gpu_training(config)

    return config


def update_input_size(config):
    """
    Convert integer image size parameters into tuples, calculate the effective input size based on image size and crop
    size, and set tiling stride if undefined.

    Args:
        config: Dictconfig: Configurable parameters object

    Returns:
        Configurable parameters with updated values

    """
    # handle image size
    if isinstance(config.transform.image_size, int):
        config.transform.image_size = (config.transform.image_size,) * 2

    if "crop_size" in config.transform.keys() and config.transform.crop_size is not None:
        if isinstance(config.transform.crop_size, int):
            config.transform.crop_size = (config.transform.crop_size,) * 2
        config.model.input_size = config.transform.crop_size
    else:
        config.model.input_size = config.transform.image_size

    if "tiling" in config.dataset.keys() and config.dataset.tiling.apply:
        if isinstance(config.dataset.tiling.tile_size, int):
            config.dataset.tiling.tile_size = (config.dataset.tiling.tile_size,) * 2
        if config.dataset.tiling.stride is None:
            config.dataset.tiling.stride = config.dataset.tiling.tile_size

    return config
