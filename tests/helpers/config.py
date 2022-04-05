from pathlib import Path
from typing import List, Optional, Union

from omegaconf import DictConfig, ListConfig

from anomalib.config import get_configurable_parameters

from .dataset import get_dataset_path


def get_test_configurable_parameters(
    dataset_path: Optional[str] = None,
    model_name: Optional[str] = None,
    model_config_path: Optional[Union[Path, str]] = None,
    weight_file: Optional[str] = None,
    openvino: bool = False,
    config_filename: Optional[str] = "config",
    config_file_extension: Optional[str] = "yaml",
) -> Union[DictConfig, ListConfig]:
    """Get configurable parameters for testing.

    Args:
        datset_path: Optional[Path]: Path to dataset
        model_name: Optional[str]:  (Default value = None)
        model_config_path: Optional[Union[Path, str]]:  (Default value = None)
        weight_file: Path to the weight file
        openvino: Use OpenVINO
        config_filename: Optional[str]:  (Default value = "config")
        config_file_extension: Optional[str]:  (Default value = "yaml")

    Returns:
        Union[DictConfig, ListConfig]: Configurable parameters in DictConfig object.
    """

    config = get_configurable_parameters(
        model_name, model_config_path, weight_file, openvino, config_filename, config_file_extension
    )

    # Update path to match the dataset path in the test image/runner
    config.dataset.path = get_dataset_path() if dataset_path is None else dataset_path

    return config
