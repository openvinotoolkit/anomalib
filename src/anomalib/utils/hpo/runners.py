"""Sweep Backends."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import gc
import logging

import torch
from lightning.pytorch.loggers import CometLogger, WandbLogger
from omegaconf import DictConfig, ListConfig, OmegaConf

from anomalib.config import update_input_size_config
from anomalib.data import get_datamodule
from anomalib.engine import Engine
from anomalib.models import get_model
from anomalib.utils.exceptions import try_import
from anomalib.utils.sweep import flatten_sweep_params, flattened_config_to_nested, set_in_nested_config

from .config import flatten_hpo_params

logger = logging.getLogger(__name__)


if try_import("wandb"):
    import wandb
if try_import("comet_ml"):
    from comet_ml import Optimizer


class WandbSweep:
    """wandb sweep.

    Args:
        project (str): Name of the project.
        sweep_config (DictConfig): Sweep configuration.
        entity (str, optional): Username or workspace to send the project to. Defaults to None.
    """

    def __init__(
        self,
        project: str,
        sweep_config: DictConfig | ListConfig,
        entity: str | None = None,
    ) -> None:
        self.sweep_config = sweep_config
        self.config: DictConfig
        self.observation_budget = sweep_config.observation_budget
        self.entity = entity
        self.project = project
        if "observation_budget" in self.sweep_config and isinstance(self.sweep_config, DictConfig):
            self.sweep_config.pop("observation_budget")

    def run(self) -> None:
        """Run the sweep."""
        flattened_hpo_params = flatten_hpo_params(self.sweep_config.parameters)
        self.config = flattened_config_to_nested(self.sweep_config.parameters)
        self.sweep_config.parameters = flattened_hpo_params
        sweep_id = wandb.sweep(
            OmegaConf.to_object(self.sweep_config),
            project=self.project,
            entity=self.entity,
        )
        wandb.agent(sweep_id, function=self.sweep, count=self.observation_budget)

    def sweep(self) -> None:
        """Load the model, update config and call fit. The metrics are logged to ```wandb``` dashboard."""
        wandb.init()
        wandb_logger = WandbLogger(config=flatten_sweep_params(self.sweep_config), log_model=False)

        for param in wandb.config.as_dict():
            set_in_nested_config(self.config, param.split("."), wandb.config[param])
        config = update_input_size_config(self.config)

        model = get_model(config.model)
        datamodule = get_datamodule(config)

        # Disable saving checkpoints as all checkpoints from the sweep will get uploaded
        engine = Engine(enable_checkpointing=False, logger=wandb_logger, devices=1)
        engine.fit(model, datamodule=datamodule)

        del model
        gc.collect()
        torch.cuda.empty_cache()


class CometSweep:
    """comet sweep.

    Args:
        project (str): Name of the project
        sweep_config (DictConfig): Sweep configuration.
        entity (str, optional): Username or workspace to send the project to. Defaults to None.
    """

    def __init__(
        self,
        project: str,
        sweep_config: DictConfig | ListConfig,
        entity: str | None = None,
    ) -> None:
        self.sweep_config = sweep_config
        self.config: DictConfig
        self.entity = entity
        self.project = project

    def run(self) -> None:
        """Run the sweep."""
        flattened_hpo_params = flatten_hpo_params(self.sweep_config.parameters)
        self.config = flattened_config_to_nested(self.sweep_config.parameters)
        self.sweep_config.parameters = flattened_hpo_params

        # comet's Optimizer takes dict as an input, not DictConfig
        std_dict = OmegaConf.to_object(self.sweep_config)

        opt = Optimizer(std_dict)

        for experiment in opt.get_experiments(project_name=self.project):
            comet_logger = CometLogger(workspace=self.entity)

            # allow pytorch-lightning to use the experiment from optimizer
            comet_logger._experiment = experiment  # noqa: SLF001
            run_params = experiment.params
            for param in run_params:
                # this check is needed as comet also returns model and sweep_config as keys
                if param in self.sweep_config.parameters:
                    set_in_nested_config(self.config, param.split("."), run_params[param])
            config = update_input_size_config(self.config)

            model = get_model(config.model)
            datamodule = get_datamodule(config)

            # Disable saving checkpoints as all checkpoints from the sweep will get uploaded
            engine = Engine(enable_checkpointing=False, logger=comet_logger, devices=1)
            engine.fit(model, datamodule=datamodule)
