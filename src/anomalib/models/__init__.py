"""Load Anomaly Model."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
import re
from importlib import import_module
from pathlib import Path

from omegaconf import DictConfig, ListConfig
from torch import load

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


def convert_snake_to_pascal_case(snake_case: str) -> str:
    """Convert snake_case to PascalCase.

    Args:
    ----
        snake_case (str): Input string in snake_case

    Returns:
        str: Output string in PascalCase

    Examples:
    --------
        >>> convert_snake_to_pascal_case("efficient_ad")
        EfficientAd

        >>> convert_snake_to_pascal_case("patchcore")
        Patchcore
    """
    return "".join(word.capitalize() for word in snake_case.split("_"))


def convert_pascal_to_snake_case(pascal_case: str) -> str:
    """Convert PascalCase to snake_case.

    Args:
    ----
        pascal_case (str): Input string in PascalCase

    Returns:
        str: Output string in snake_case

    Examples:
    --------
        >>> convert_pascal_to_snake_case("EfficientAd")
        efficient_ad

        >>> convert_pascal_to_snake_case("Patchcore")
        patchcore
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", pascal_case).lower()


def get_available_models() -> list[str]:
    """Get list of available models.

    Returns:
        list[str]: List of available models.

    Example:
    -------
        >>> get_available_models()
        ['ai_vad', 'cfa', 'cflow', 'csflow', 'dfkde', 'dfm', 'draem', 'efficient_ad', 'fastflow', ...]
    """
    return [convert_pascal_to_snake_case(cls.__name__) for cls in AnomalyModule.__subclasses__()]


def get_model(config: DictConfig | ListConfig) -> AnomalyModule:
    """Load model from the configuration file.

    Works only when the convention for model naming is followed.

    The convention for writing model classes is
    `anomalib.models.<model_name>.lightning_model.<ModelName>Lightning`
    `anomalib.models.stfpm.lightning_model.StfpmLightning`

    Args:
    ----
        config (DictConfig | ListConfig): Config.yaml loaded using OmegaConf

    Raises:
    ------
        ValueError: If unsupported model is passed

    Returns:
        AnomalyModule: Anomaly Model
    """
    logger.info("Loading the model.")
    model: AnomalyModule

    try:
        module = import_module(".".join(config.model.class_path.split(".")[:-1]))
        model = getattr(module, config.model.class_path.split(".")[-1])
        model = model(**config.model.init_args)
    except ModuleNotFoundError:
        logger.exception("Could not find the model class: %s", config.model.class_path)
        raise

    if "init_weights" in config and config.init_weights:
        model.load_state_dict(load(str(Path(config.project.path) / config.init_weights))["state_dict"], strict=False)

    return model
