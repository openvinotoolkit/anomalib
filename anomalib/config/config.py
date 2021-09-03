"""
Configurable Getter
"""

from pathlib import Path
from typing import Optional, Union

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

    if weight_file:
        config.weight_file = weight_file

    # NNCF Parameters
    config = update_config_for_nncf(config)

    config.openvino = openvino
    if openvino:
        config.trainer.gpus = 0

    return config


def update_input_size(config):
    """
    Convert integer image size parameters into tuples and calculate the effective input size based on image size,
    crop size and tile size.

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

    if "tiling" in config.dataset.keys() and config.dataset.tiling.apply:
        config.model.input_size = (config.dataset.tiling.tile_size,) * 2
        if config.dataset.tiling.stride is None:
            config.dataset.tiling.stride = config.dataset.tiling.tile_size
    elif "crop_size" in config.transform.keys() and config.transform.crop_size is not None:
        config.model.input_size = config.transform.crop_size
    else:
        config.model.input_size = config.transform.image_size
    return config
