"""Sweep Backends."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union

import pytorch_lightning as pl
from comet_ml import Optimizer
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.loggers import CometLogger, WandbLogger

import wandb
from anomalib.config import update_input_size_config
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.sweep import (
    flatten_sweep_params,
    get_sweep_callbacks,
    set_in_nested_config,
)

from .config import flatten_hpo_params


class WandbSweep:
    """wandb sweep.

    Args:
        config (DictConfig): Original model configuration.
        sweep_config (DictConfig): Sweep configuration.
        entity (str, optional): Username or workspace to send the project to. Defaults to None.
    """

    def __init__(
        self,
        config: Union[DictConfig, ListConfig],
        sweep_config: Union[DictConfig, ListConfig],
        entity: Optional[str] = None,
    ) -> None:
        self.config = config
        self.sweep_config = sweep_config
        self.observation_budget = sweep_config.observation_budget
        self.entity = entity
        if "observation_budget" in self.sweep_config.keys():
            # this instance check is to silence mypy.
            if isinstance(self.sweep_config, DictConfig):
                self.sweep_config.pop("observation_budget")

    def run(self):
        """Run the sweep."""
        flattened_hpo_params = flatten_hpo_params(self.sweep_config.parameters)
        self.sweep_config.parameters = flattened_hpo_params
        sweep_id = wandb.sweep(
            OmegaConf.to_object(self.sweep_config),
            project=f"{self.config.model.name}_{self.config.dataset.name}",
            entity=self.entity,
        )
        wandb.agent(sweep_id, function=self.sweep, count=self.observation_budget)

    def sweep(self):
        """Method to load the model, update config and call fit. The metrics are logged to ```wandb``` dashboard."""
        wandb_logger = WandbLogger(config=flatten_sweep_params(self.sweep_config), log_model=False)
        sweep_config = wandb_logger.experiment.config

        for param in sweep_config.keys():
            set_in_nested_config(self.config, param.split("."), sweep_config[param])
        config = update_input_size_config(self.config)

        model = get_model(config)
        datamodule = get_datamodule(config)
        callbacks = get_sweep_callbacks(config)

        # Disable saving checkpoints as all checkpoints from the sweep will get uploaded
        config.trainer.checkpoint_callback = False

        trainer = pl.Trainer(**config.trainer, logger=wandb_logger, callbacks=callbacks)
        trainer.fit(model, datamodule=datamodule)


class CometSweep:
    """comet sweep.

    Args:
        config (DictConfig): Original model configuration.
        sweep_config (DictConfig): Sweep configuration.
        entity (str, optional): Username or workspace to send the project to. Defaults to None.
    """

    def __init__(
        self,
        config: Union[DictConfig, ListConfig],
        sweep_config: Union[DictConfig, ListConfig],
        entity: Optional[str] = None,
    ) -> None:
        self.config = config
        self.sweep_config = sweep_config
        self.entity = entity

    def run(self):
        """Run the sweep."""
        flattened_hpo_params = flatten_hpo_params(self.sweep_config.parameters)
        self.sweep_config.parameters = flattened_hpo_params

        # comet's Optimizer takes dict as an input, not DictConfig
        std_dict = OmegaConf.to_object(self.sweep_config)

        opt = Optimizer(std_dict)

        project_name = f"{self.config.model.name}_{self.config.dataset.name}"

        for experiment in opt.get_experiments(project_name=project_name):
            comet_logger = CometLogger(workspace=self.entity)

            # allow pytorch-lightning to use the experiment from optimizer
            comet_logger._experiment = experiment  # pylint: disable=protected-access
            run_params = experiment.params
            for param in run_params.keys():
                # this check is needed as comet also returns model and sweep_config as keys
                if param in self.sweep_config.parameters.keys():
                    set_in_nested_config(self.config, param.split("."), run_params[param])
            config = update_input_size_config(self.config)

            model = get_model(config)
            datamodule = get_datamodule(config)
            callbacks = get_sweep_callbacks(config)

            # Disable saving checkpoints as all checkpoints from the sweep will get uploaded
            config.trainer.checkpoint_callback = False

            trainer = pl.Trainer(**config.trainer, logger=comet_logger, callbacks=callbacks)
            trainer.fit(model, datamodule=datamodule)
