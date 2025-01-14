"""Callbacks for Anomalib models.

This module provides various callbacks used in Anomalib for model training, logging, and optimization.
The callbacks include model checkpointing, graph logging, model loading, tiler configuration, and timing.

The module exports the following callbacks:

- :class:`ModelCheckpoint`: Save model checkpoints during training
- :class:`GraphLogger`: Log model computation graphs
- :class:`LoadModelCallback`: Load pre-trained model weights
- :class:`TilerConfigurationCallback`: Configure image tiling settings
- :class:`TimerCallback`: Track training/inference timing

Example:
    Get default callbacks based on configuration:

    >>> from anomalib.callbacks import get_callbacks
    >>> from omegaconf import DictConfig
    >>> config = DictConfig({"trainer": {}, "project": {"path": "/tmp"}})
    >>> callbacks = get_callbacks(config)
    >>> isinstance(callbacks, list)
    True

    Use callbacks in trainer:

    >>> import lightning.pytorch as pl
    >>> trainer = pl.Trainer(callbacks=callbacks)

See Also:
    - :mod:`anomalib.callbacks.checkpoint`: Model checkpoint callback
    - :mod:`anomalib.callbacks.graph`: Graph logging callback
    - :mod:`anomalib.callbacks.model_loader`: Model loading callback
    - :mod:`anomalib.callbacks.tiler_configuration`: Tiler configuration callback
    - :mod:`anomalib.callbacks.timer`: Timer callback
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from importlib import import_module
from pathlib import Path

import yaml
from jsonargparse import Namespace
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, ListConfig, OmegaConf

from .checkpoint import ModelCheckpoint
from .graph import GraphLogger
from .model_loader import LoadModelCallback
from .tiler_configuration import TilerConfigurationCallback
from .timer import TimerCallback

__all__ = [
    "ModelCheckpoint",
    "GraphLogger",
    "LoadModelCallback",
    "TilerConfigurationCallback",
    "TimerCallback",
]


logger = logging.getLogger(__name__)


def get_callbacks(config: DictConfig | ListConfig | Namespace) -> list[Callback]:
    """Get default callbacks for Anomalib models based on configuration.

    This function returns a list of callbacks based on the provided configuration.
    It automatically adds:

    - Model loading callback if checkpoint path is specified
    - NNCF optimization callback if NNCF optimization is enabled

    Args:
        config (DictConfig | ListConfig | Namespace): Configuration object containing model and training settings.
            Expected to have the following structure:

            .. code-block:: yaml

                trainer:
                    ckpt_path: Optional[str]  # Path to model checkpoint
                optimization:
                    nncf:
                        apply: bool  # Whether to apply NNCF optimization
                        # Other NNCF config options
                project:
                    path: str  # Project directory path

    Returns:
        list[Callback]: List of PyTorch Lightning callbacks to be used during training.
            May include:

            - :class:`LoadModelCallback`: For loading model checkpoints
            - :class:`NNCFCallback`: For neural network compression
            - Other default callbacks

    Example:
        >>> from omegaconf import DictConfig
        >>> config = DictConfig({
        ...     "trainer": {"ckpt_path": None},
        ...     "project": {"path": "/tmp"},
        ...     "optimization": {"nncf": {"apply": False}}
        ... })
        >>> callbacks = get_callbacks(config)
        >>> isinstance(callbacks, list)
        True

    Note:
        NNCF is imported dynamically only when required since it conflicts with
        some kornia JIT operations.
    """
    logger.info("Loading the callbacks")

    callbacks: list[Callback] = []

    if "ckpt_path" in config.trainer and config.ckpt_path is not None:
        load_model = LoadModelCallback(config.ckpt_path)
        callbacks.append(load_model)

    if "optimization" in config and "nncf" in config.optimization and config.optimization.nncf.apply:
        # NNCF wraps torch's jit which conflicts with kornia's jit calls.
        # Hence, nncf is imported only when required
        nncf_module = import_module("anomalib.utils.callbacks.nncf.callback")
        nncf_callback = nncf_module.NNCFCallback
        nncf_config = yaml.safe_load(OmegaConf.to_yaml(config.optimization.nncf))
        callbacks.append(
            nncf_callback(
                config=nncf_config,
                export_dir=str(Path(config.project.path) / "compressed"),
            ),
        )

    return callbacks
