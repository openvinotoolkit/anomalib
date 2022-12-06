"""Configure Optimizers.

LightningCLI adds optimizer to lightning class automatically. This function is used to configure the optimizer from the
config file for entry point scripts.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

from omegaconf import DictConfig

from anomalib.models.components import AnomalyModule


def configure_optimizer(model: AnomalyModule, config: DictConfig) -> None:
    """Adds optmizer to the Lightning module.

    Args:
        model (AnomalyModule): Lightning module
        config (DictConfig): Config.yaml loaded using OmegaConf
    """
    if "optimizer" in config:
        optimizer_module = import_module(".".join(config.optimizer.class_path.split(".")[:-1]))
        optimizer_class = getattr(optimizer_module, config.optimizer.class_path.split(".")[-1])
        optimizer = optimizer_class(params=model.model.parameters(), **config.optimizer.init_args)
        model.configure_optimizers = lambda: optimizer
