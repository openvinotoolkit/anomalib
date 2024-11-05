"""Config upgrade tool.

This module provides a tool for migrating Anomalib configuration files from
v0.* format to v1.* format. The `ConfigAdapter` class in this module is
responsible for migrating different sections of the configuration file.

Example:
    # Create a ConfigAdapter instance with the path to the old config file
    adapter = ConfigAdapter("/path/to/old_config.yaml")

    # Upgrade the configuration to v1 format
    upgraded_config = adapter.upgrade_all()

    # Save the upgraded configuration to a new file
    adapter.save_config(upgraded_config, "/path/to/new_config.yaml")
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib
import inspect
from pathlib import Path
from typing import Any

import yaml

from anomalib.models import convert_snake_to_pascal_case
from anomalib.utils.config import to_tuple


def get_class_signature(module_path: str, class_name: str) -> inspect.Signature:
    """Get the signature of a class constructor.

    Args:
        module_path (str): The path to the module containing the class.
        class_name (str): The name of the class.

    Returns:
        inspect.Signature: The signature of the class constructor.

    Examples:
        >>> get_class_signature('my_module', 'MyClass')
        <Signature (self, arg1, arg2=None)>

        >>> get_class_signature('other_module', 'OtherClass')
        <Signature (self, arg1, arg2, *, kwarg1='default')>
    """
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return inspect.signature(cls.__init__)


def get_class_init_args(module_path: str, class_name: str) -> dict[str, Any | None]:
    """Get the initialization arguments of a class.

    Args:
        module_path (str): The path of the module containing the class.
        class_name (str): The name of the class.

    Returns:
        dict[str, Any | None]: A dictionary containing the initialization arguments
            of the class, with argument names as keys and default values as values.

    Example:
        >>> get_class_init_args("my_module", "MyClass")
        {'arg1': None, 'arg2': 0, 'arg3': 'default'}
    """
    init_signature = get_class_signature(module_path, class_name)
    return {
        k: v.default if v.default is not inspect.Parameter.empty else None
        for k, v in init_signature.parameters.items()
        if k != "self"
    }


def overwrite_args(
    default_args: dict[str, Any],
    new_args: dict[str, Any],
    excluded_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Overwrite the default arguments with the new arguments.

    Args:
        default_args (dict[str, Any]): The default arguments.
        new_args (dict[str, Any]): The new arguments.
        excluded_keys (list[str] | None, optional): A list of keys to exclude
            from the new arguments.
            Defaults to ``None``.

    Returns:
        dict[str, Any]: The updated arguments.

    Example:
        Overwrite the default arguments with the new arguments
        >>> default_args = {"a": 1, "b": 2, "c": 3}
        >>> new_args = {"b": 4, "c": 5}
        >>> updated_args = overwrite_args(default_args, new_args)
        >>> print(updated_args)
        Output: {"a": 1, "b": 4, "c": 5}
    """
    if excluded_keys is None:
        excluded_keys = []

    for key, value in new_args.items():
        if key in default_args and key not in excluded_keys:
            default_args[key] = value

    return default_args


class ConfigAdapter:
    """Class responsible for migrating configuration data."""

    def __init__(self, config: str | Path | dict[str, Any]) -> None:
        self.old_config = self.safe_load(config) if isinstance(config, str | Path) else config

    @staticmethod
    def safe_load(path: str | Path) -> dict:
        """Load a yaml file and return the content as a dictionary."""
        with Path(path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def upgrade_data_config(self) -> dict[str, Any]:
        """Upgrade data config."""
        # Get the dataset class name based on the format in the old config
        dataset_class_name = convert_snake_to_pascal_case(self.old_config["dataset"]["format"])

        # mvtec has an exception and is written as MVTec. Convert all Mvtec datasets to MVTec
        dataset_class_name = dataset_class_name.replace("Mvtec", "MVTec")

        # Get the class path and init args.
        class_path = f"anomalib.data.{dataset_class_name}"
        init_args = get_class_init_args("anomalib.data", dataset_class_name)

        # Replace the old config key ``path`` with ``root``
        if "path" in self.old_config["dataset"]:
            self.old_config["dataset"]["root"] = self.old_config["dataset"].pop("path")

        # Overwrite the init_args with the old config
        init_args = overwrite_args(
            init_args,
            self.old_config["dataset"],
        )

        # Input size is a list in the old config, convert it to a tuple
        init_args["image_size"] = to_tuple(init_args["image_size"])

        return {
            "data": {
                "class_path": class_path,
                "init_args": init_args,
            },
        }

    def upgrade_model_config(self) -> dict[str, Any]:
        """Upgrade the model config to v1 format."""
        # Get the model class name
        model_name = convert_snake_to_pascal_case(self.old_config["model"]["name"])

        # Get the models args.
        init_args = get_class_init_args("anomalib.models", model_name)

        # Overwrite the init_args with the old config
        init_args = overwrite_args(
            init_args,
            self.old_config["model"],
            excluded_keys=["name", "early_stopping", "normalization_method"],
        )

        return {
            "model": {
                "class_path": f"anomalib.models.{model_name}",
                "init_args": init_args,
            },
        }

    def upgrade_normalization_config(self) -> dict[str, Any]:
        """Upgrade the normalization config to v1 format."""
        return {"normalization": {"normalization_method": self.old_config["model"]["normalization_method"]}}

    def upgrade_metrics_config(self) -> dict[str, Any]:
        """Upgrade the metrics config to v1 format, with streamlined logic."""
        # Define a direct mapping for threshold methods to class names
        threshold_class_map = {
            "adaptive": "F1AdaptiveThreshold",
            "manual": "ManualThreshold",
        }

        threshold_method = self.old_config.get("metrics", {}).get("threshold", {}).get("method")
        class_name = threshold_class_map.get(threshold_method)

        if not class_name:
            msg = f"Unknown threshold method {threshold_method}. Available methods are 'adaptive' or 'manual'."
            raise ValueError(msg)

        new_config: dict[str, Any] = {
            "metrics": {
                "image": self.old_config.get("metrics", {}).get("image"),
                "pixel": self.old_config.get("metrics", {}).get("pixel"),
                "threshold": {
                    "class_path": f"anomalib.metrics.{class_name}",
                    "init_args": {"default_value": 0.5},
                },
            },
        }

        return new_config

    def upgrade_visualization_config(self) -> dict[str, Any]:
        """Upgrade the visualization config to v1 format."""
        # Initialize the new configuration with default values from the new format
        new_config = {
            "visualization": {
                "visualizers": None,
                "save": False,
                "log": False,
                "show": False,
            },
        }

        # Map old configuration values to the new format
        if "visualization" in self.old_config:
            old_config = self.old_config["visualization"]

            # Set new configuration values based on the old configuration
            new_config["visualization"]["save"] = old_config.get("save_images", False)
            new_config["visualization"]["log"] = old_config.get("log_images", False)
            new_config["visualization"]["show"] = old_config.get("show_images", False)

        return new_config

    def upgrade_logging_config(self) -> dict[str, Any]:
        """Upgrade logging config to v1 format."""
        return {"logging": {"log_graph": self.old_config["logging"]["log_graph"]}}

    def add_results_dir_config(self) -> dict[str, Any]:
        """Create results_dir field in v1 config."""
        return {
            "results_dir": {
                "path": self.old_config["project"]["path"],
                "unique": False,
            },
        }

    def add_seed_config(self) -> dict[str, Any]:
        """Create seed everything field in v1 config."""
        return {"seed_everything": bool(self.old_config["project"]["seed"])}

    @staticmethod
    def add_ckpt_path_config() -> dict[str, Any]:
        """Create checkpoint path directory in v1 config."""
        return {"ckpt_path": None}

    def add_task_config(self) -> dict[str, str]:
        """Create task field in v1 config."""
        return {"task": self.old_config["dataset"]["task"]}

    def upgrade_trainer_config(self) -> dict[str, Any]:
        """Upgrade Trainer config to v1 format."""
        # Get the signature of the Trainer class's __init__ method
        init_args = get_class_init_args("lightning.pytorch", "Trainer")

        # Overwrite the init_args with the old config
        init_args = overwrite_args(init_args, self.old_config["trainer"], excluded_keys=["strategy"])

        # Early stopping callback was passed to model config in v0.*
        if "early_stopping" in self.old_config.get("model", {}):
            early_stopping_config = {
                "class_path": "lightning.pytorch.callbacks.EarlyStopping",
                "init_args": self.old_config["model"]["early_stopping"],
            }

            # Rename metric to monitor
            if "metric" in early_stopping_config["init_args"]:
                early_stopping_config["init_args"]["monitor"] = early_stopping_config["init_args"].pop("metric")

            if init_args["callbacks"] is None:
                init_args["callbacks"] = [early_stopping_config]
            else:
                init_args["callbacks"].append(early_stopping_config)

        return {"trainer": init_args}

    def upgrade_all(self) -> dict[str, Any]:
        """Upgrade Anomalib v0.* config to v1 config format."""
        new_config = {}

        new_config.update(self.upgrade_data_config())
        new_config.update(self.upgrade_model_config())
        new_config.update(self.upgrade_normalization_config())
        new_config.update(self.upgrade_metrics_config())
        new_config.update(self.upgrade_visualization_config())
        new_config.update(self.upgrade_logging_config())
        new_config.update(self.add_seed_config())
        new_config.update(self.add_task_config())
        new_config.update(self.add_results_dir_config())
        new_config.update(self.add_ckpt_path_config())
        new_config.update(self.upgrade_trainer_config())

        return new_config

    @staticmethod
    def save_config(config: dict, path: str | Path) -> None:
        """Save the given configuration dictionary to a YAML file.

        Args:
            config (dict): The configuration dictionary to be saved.
            path (str | Path): The path to the output file.

        Returns:
            None
        """
        with Path(path).open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False)


def get_args() -> argparse.Namespace:
    """Get the command line arguments."""
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Upgrade configuration files from v0.* format to v1.* format.")
    parser.add_argument("-i", "--input_config", type=Path, required=True, help="Path to the old configuration file.")
    parser.add_argument("-o", "--output_config", type=Path, required=True, help="Path to the new configuration file.")

    # Parse arguments
    args = parser.parse_args()

    # Ensure the provided paths are valid
    if not args.input_config.exists():
        msg = f"The specified old configuration file does not exist: {args.input_config}"
        raise FileNotFoundError(msg)

    return args


def upgrade(old_config_path: Path, new_config_path: Path) -> None:
    """Upgrade Anomalib configuration file from v0.* to v1.* format.

    Args:
        old_config_path (Path): Path to the old configuration file.
        new_config_path (Path): Path to the new configuration file.
    """
    config_adapter = ConfigAdapter(config=old_config_path)
    new_config = config_adapter.upgrade_all()
    config_adapter.save_config(new_config, new_config_path)


if __name__ == "__main__":
    args = get_args()
    upgrade(args.input_config, args.output_config)
