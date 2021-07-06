"""
Configurable Getter
"""

from pathlib import Path
from typing import Optional, Union

from omegaconf import DictConfig, ListConfig, OmegaConf


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

    image_size = config.dataset.image_size
    if isinstance(image_size, int):
        config.dataset.image_size = (image_size, image_size)

    if "crop_size" in config.dataset.keys():
        if isinstance(config.dataset.crop_size, int):
            config.dataset.crop_size = (config.dataset.crop_size,) * 2
    else:
        config.dataset.crop_size = config.dataset.image_size

    # Project Configs
    project_path = Path(config.project.path) / config.model.name / config.dataset.name / config.dataset.category
    (project_path / "weights").mkdir(parents=True, exist_ok=True)
    (project_path / "images").mkdir(parents=True, exist_ok=True)
    config.project.path = str(project_path)

    if weight_file:
        config.weight_file = weight_file

    # NNCF Parameters
    crop_size = config.dataset.crop_size
    sample_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
    if "optimization" in config.keys():
        if "nncf" in config.optimization.keys():
            config.optimization.nncf.input_info.sample_size = [1, 3, *sample_size]

    config.openvino = openvino
    if openvino:
        config.trainer.gpus = 0

    return config
