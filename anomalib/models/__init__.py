"""Load Anomaly Model."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from importlib import import_module
from typing import Union

from omegaconf import DictConfig, ListConfig
from torch import load

from anomalib.models.cflow import Cflow
from anomalib.models.components import AnomalyModule
from anomalib.models.dfkde import Dfkde
from anomalib.models.dfm import Dfm
from anomalib.models.draem import Draem
from anomalib.models.fastflow import Fastflow
from anomalib.models.ganomaly import Ganomaly
from anomalib.models.padim import Padim
from anomalib.models.patchcore import Patchcore
from anomalib.models.reverse_distillation import ReverseDistillation
from anomalib.models.stfpm import Stfpm

__all__ = [
    "Cflow",
    "Dfkde",
    "Dfm",
    "Draem",
    "Fastflow",
    "Ganomaly",
    "Padim",
    "Patchcore",
    "ReverseDistillation",
    "Stfpm",
]

logger = logging.getLogger(__name__)


def get_model(config: Union[DictConfig, ListConfig]) -> AnomalyModule:
    """Load model from the configuration file.

    Works only when the convention for model naming is followed.

    The convention for writing model classes is
    `anomalib.models.<model_name>.lightning_model.<ModelName>Lightning`
    `anomalib.models.stfpm.lightning_model.StfpmLightning`

    Args:
        config (Union[DictConfig, ListConfig]): Config.yaml loaded using OmegaConf

    Raises:
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
    except ModuleNotFoundError as exception:
        logger.error("Could not find the model class: %s", config.model.class_path)
        raise exception

    if config.trainer.resume_from_checkpoint is not None:
        model.load_state_dict(load(config.trainer.resume_from_checkpoint)["state_dict"], strict=False)

    return model
