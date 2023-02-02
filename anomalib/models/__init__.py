"""Load Anomaly Model."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from importlib import import_module

from omegaconf import DictConfig, ListConfig
from torch import load

from anomalib.models.cfa import Cfa
from anomalib.models.cflow import Cflow
from anomalib.models.components import AnomalyModule
from anomalib.models.csflow import Csflow
from anomalib.models.dfkde import Dfkde
from anomalib.models.dfm import Dfm
from anomalib.models.draem import Draem
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
]

logger = logging.getLogger(__name__)


def _snake_to_pascal_case(model_name: str) -> str:
    """Convert model name from snake case to Pascal case.

    Args:
        model_name (str): Model name in snake case.

    Returns:
        str: Model name in Pascal case.
    """
    return "".join([split.capitalize() for split in model_name.split("_")])


def get_model(config: DictConfig | ListConfig) -> AnomalyModule:
    """Load model from the configuration file.

    Works only when the convention for model naming is followed.

    The convention for writing model classes is
    `anomalib.models.<model_name>.lightning_model.<ModelName>Lightning`
    `anomalib.models.stfpm.lightning_model.StfpmLightning`

    Args:
        config (DictConfig | ListConfig): Config.yaml loaded using OmegaConf

    Raises:
        ValueError: If unsupported model is passed

    Returns:
        AnomalyModule: Anomaly Model
    """
    logger.info("Loading the model.")

    model_list: list[str] = [
        "cfa",
        "cflow",
        "csflow",
        "dfkde",
        "dfm",
        "draem",
        "fastflow",
        "ganomaly",
        "padim",
        "patchcore",
        "reverse_distillation",
        "rkde",
        "stfpm",
    ]
    model: AnomalyModule

    if config.model.name in model_list:
        module = import_module(f"anomalib.models.{config.model.name}")
        model = getattr(module, f"{_snake_to_pascal_case(config.model.name)}Lightning")(config)

    else:
        raise ValueError(f"Unknown model {config.model.name}!")

    if "init_weights" in config.keys() and config.init_weights:
        model.load_state_dict(load(os.path.join(config.project.path, config.init_weights))["state_dict"], strict=False)

    return model
