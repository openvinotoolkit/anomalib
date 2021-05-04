from pathlib import Path
from typing import Optional, Union

from omegaconf import OmegaConf


def get_configurable_parameters(
    model_name: Optional[str] = None,
    model_config_path: Optional[Union[Path, str]] = None,
    config_filename: Optional[str] = "config",
    config_file_extension: Optional[str] = "yaml",
):
    if model_name is None and model_config_path is None:
        raise ValueError(
            "Both model_name and model config path cannot be None! "
            "Please provide a model name or path to a config file!"
        )

    if model_config_path is None:
        model_config_path = Path(f"anomalib/models/{model_name}/{config_filename}.{config_file_extension}")

    config = OmegaConf.load(model_config_path)
    return config
