"""Load Anomaly Model."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import inspect
import logging
import re
from importlib import import_module

from jsonargparse import Namespace
from omegaconf import DictConfig, ListConfig

from anomalib.models.ai_vad import AiVad
from anomalib.models.cfa import Cfa
from anomalib.models.cflow import Cflow
from anomalib.models.components import AnomalyModule
from anomalib.models.csflow import Csflow
from anomalib.models.dfkde import Dfkde
from anomalib.models.dfm import Dfm
from anomalib.models.draem import Draem
from anomalib.models.efficient_ad import EfficientAd
from anomalib.models.fastflow import Fastflow
from anomalib.models.ganomaly import Ganomaly
from anomalib.models.padim import Padim
from anomalib.models.patchcore import Patchcore
from anomalib.models.reverse_distillation import ReverseDistillation
from anomalib.models.rkde import Rkde
from anomalib.models.stfpm import Stfpm


class UnknownModelError(ModuleNotFoundError):
    ...


__all__ = [
    "Cfa",
    "Cflow",
    "Csflow",
    "Dfkde",
    "Dfm",
    "Draem",
    "Fastflow",
    "Ganomaly",
    "Padim",
    "Patchcore",
    "ReverseDistillation",
    "Rkde",
    "Stfpm",
    "AiVad",
    "EfficientAd",
]

logger = logging.getLogger(__name__)


def _convert_pascal_to_snake_case(pascal_case: str) -> str:
    """Convert PascalCase to snake_case.

    Args:
        pascal_case (str): Input string in PascalCase

    Returns:
        str: Output string in snake_case

    Examples:
        >>> _convert_pascal_to_snake_case("EfficientAd")
        efficient_ad

        >>> _convert_pascal_to_snake_case("Patchcore")
        patchcore
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", pascal_case).lower()


def get_available_models() -> list[str]:
    """Get list of available models.

    Returns:
        list[str]: List of available models.

    Example:
        >>> get_available_models()
        ['ai_vad', 'cfa', 'cflow', 'csflow', 'dfkde', 'dfm', 'draem', 'efficient_ad', 'fastflow', ...]
    """
    return [_convert_pascal_to_snake_case(cls.__name__) for cls in AnomalyModule.__subclasses__()]


def _get_model_by_name(name: str) -> AnomalyModule:
    """Get's the model by name.

    Args:
        name (str): Name of the model. The name is case insensitive.

    Raises:
        ModuleNotFoundError: If unsupported model is passed

    Returns:
        AnomalyModule: Anomaly Model
    """
    logger.info("Loading the model.")
    model: AnomalyModule | None = None

    # search for model by name in available models.
    for model_name in get_available_models():
        if name.lower() in model_name:
            module = import_module(f"anomalib.models.{model_name}")
            for class_name in dir(module):
                if class_name.lower() in class_name.lower():
                    model_class = getattr(module, class_name, None)
                    if inspect.isclass(model_class):
                        model = model_class()
                        break
            break
    if model is None:
        logger.exception(f"Could not find the model {name}. Available models are {get_available_models()}")
        raise UnknownModelError

    return model


def get_model(config: DictConfig | ListConfig | str | Namespace) -> AnomalyModule:
    """Get Anomaly Model.

    Args:
        config (DictConfig | ListConfig | str): Can either be a configuration or a string.

    Examples:
        >>> get_model("Padim")
        >>> get_model({"class_path": "Padim"})
        >>> get_model({"class_path": "Padim", "init_args": {"input_size": (100, 100)}})
        >>> get_model({"class_path": "anomalib.models.Padim", "init_args": {"input_size": (100, 100)}}})

    Raises:
        TypeError: If unsupported type is passed.

    Returns:
        AnomalyModule: Anomaly Model
    """
    model: AnomalyModule
    if isinstance(config, str):
        model = _get_model_by_name(config)
    elif isinstance(config, DictConfig | ListConfig | Namespace):
        if len(config.class_path.split(".")) > 1:
            module = import_module(".".join(config.class_path.split(".")[:-1]))
        else:
            module = import_module(f"anomalib.models.{config.class_path}")
        model_class = getattr(module, config.class_path.split(".")[-1])
        model = model_class(**config.get("init_args", {}))
    else:
        logger.error(f"Unsupported type {type(config)} for model configuration.")
        raise TypeError
    return model
