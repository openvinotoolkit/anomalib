"""Load Anomaly Model."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from importlib import import_module

from jsonargparse import Namespace
from omegaconf import DictConfig, OmegaConf

from anomalib.models.components import AnomalyModule
from anomalib.utils.path import convert_to_snake_case

from .image import (
    Cfa,
    Cflow,
    Csflow,
    Dfkde,
    Dfm,
    Draem,
    Dsr,
    EfficientAd,
    Fastflow,
    Fre,
    Ganomaly,
    Padim,
    Patchcore,
    ReverseDistillation,
    Rkde,
    Stfpm,
    Uflow,
    WinClip,
)
from .video import AiVad


class UnknownModelError(ModuleNotFoundError):
    pass


__all__ = [
    "Cfa",
    "Cflow",
    "Csflow",
    "Dfkde",
    "Dfm",
    "Draem",
    "Dsr",
    "EfficientAd",
    "Fastflow",
    "Fre",
    "Ganomaly",
    "Padim",
    "Patchcore",
    "ReverseDistillation",
    "Rkde",
    "Stfpm",
    "Uflow",
    "AiVad",
    "WinClip",
]

logger = logging.getLogger(__name__)


def convert_snake_to_pascal_case(snake_case: str) -> str:
    """Convert snake_case to PascalCase.

    Args:
        snake_case (str): Input string in snake_case

    Returns:
        str: Output string in PascalCase

    Examples:
        >>> _convert_snake_to_pascal_case("efficient_ad")
        EfficientAd

        >>> _convert_snake_to_pascal_case("patchcore")
        Patchcore
    """
    return "".join(word.capitalize() for word in snake_case.split("_"))


def get_available_models() -> set[str]:
    """Get set of available models.

    Returns:
        set[str]: List of available models.

    Example:
        >>> get_available_models()
        ['ai_vad', 'cfa', 'cflow', 'csflow', 'dfkde', 'dfm', 'draem', 'efficient_ad', 'fastflow', ...]
    """
    return {convert_to_snake_case(cls.__name__) for cls in AnomalyModule.__subclasses__()}


def _get_model_class_by_name(name: str) -> type[AnomalyModule]:
    """Retrieves an anomaly model based on its name.

    Args:
        name (str): The name of the model to retrieve. The name is case insensitive.

    Raises:
        UnknownModelError: If the model is not found.

    Returns:
        type[AnomalyModule]: Anomaly Model
    """
    logger.info("Loading the model.")
    model_class: type[AnomalyModule] | None = None

    name = convert_snake_to_pascal_case(name).lower()
    for model in AnomalyModule.__subclasses__():
        if name == model.__name__.lower():
            model_class = model
    if model_class is None:
        logger.exception(f"Could not find the model {name}. Available models are {get_available_models()}")
        raise UnknownModelError

    return model_class


def get_model(model: DictConfig | str | dict | Namespace, *args, **kwdargs) -> AnomalyModule:
    """Get Anomaly Model.

    Args:
        model (DictConfig | str): Can either be a configuration or a string.
        *args: Variable length argument list for model init.
        **kwdargs: Arbitrary keyword arguments for model init.

    Examples:
        >>> get_model("Padim")
        >>> get_model("efficient_ad")
        >>> get_model("Patchcore", input_size=(100, 100))
        >>> get_model({"class_path": "Padim"})
        >>> get_model({"class_path": "Patchcore"}, input_size=(100, 100))
        >>> get_model({"class_path": "Padim", "init_args": {"input_size": (100, 100)}})
        >>> get_model({"class_path": "anomalib.models.Padim", "init_args": {"input_size": (100, 100)}}})

    Raises:
        TypeError: If unsupported type is passed.

    Returns:
        AnomalyModule: Anomaly Model
    """
    _model: AnomalyModule
    if isinstance(model, str):
        _model_class = _get_model_class_by_name(model)
        _model = _model_class(*args, **kwdargs)
    elif isinstance(model, DictConfig | Namespace | dict):
        if isinstance(model, dict):
            model = OmegaConf.create(model)
        try:
            if len(model.class_path.split(".")) > 1:
                module = import_module(".".join(model.class_path.split(".")[:-1]))
            else:
                module = import_module("anomalib.models")
        except ModuleNotFoundError as exception:
            logger.exception(
                f"Could not find the module {model.class_path}. Available models are {get_available_models()}",
            )
            raise UnknownModelError from exception
        try:
            model_class = getattr(module, model.class_path.split(".")[-1])
            init_args = model.get("init_args", {})
            if len(kwdargs) > 0:
                init_args.update(kwdargs)
            _model = model_class(*args, **init_args)
        except AttributeError as exception:
            logger.exception(
                f"Could not find the model {model.class_path}. Available models are {get_available_models()}",
            )
            raise UnknownModelError from exception
    else:
        logger.error(f"Unsupported type {type(model)} for model configuration.")
        raise TypeError
    return _model
