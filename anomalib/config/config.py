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

    # Add project path.
    project_path = Path(config.project.path) / config.model.name / config.dataset.name / config.dataset.category
    (project_path / "weights").mkdir(parents=True, exist_ok=True)
    (project_path / "images").mkdir(parents=True, exist_ok=True)
    config.project.path = str(project_path)

    if weight_file:
        config.weight_file = weight_file

    config.openvino = openvino
    if openvino:
        config.trainer.gpus = 0

    return config
