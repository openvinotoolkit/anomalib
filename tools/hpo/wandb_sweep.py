"""Run wandb sweep."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from utils import flatten_hpo_params

import wandb
from anomalib.config import get_configurable_parameters, update_input_size_config
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.sweep import flatten_sweep_params, set_in_nested_config


class WandbSweep:
    """wandb sweep.

    Args:
        config (DictConfig): Original model configuration.
        sweep_config (DictConfig): Sweep configuration.
    """

    def __init__(self, config: DictConfig, sweep_config: DictConfig) -> None:
        self.config = config
        self.sweep_config = sweep_config
        self.observation_budget = sweep_config.observation_budget
        if "observation_budget" in self.sweep_config.keys():
            self.sweep_config.pop("observation_budget")

    def run(self):
        """Run the sweep."""
        flattened_hpo_params = flatten_hpo_params(self.sweep_config.parameters)
        self.sweep_config.parameters = flattened_hpo_params
        sweep_id = wandb.sweep(
            OmegaConf.to_object(self.sweep_config),
            project=f"{self.config.model.name}_{self.config.dataset.name}",
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

        # Disable saving checkpoints as all checkpoints from the sweep will get uploaded
        config.trainer.checkpoint_callback = False

        trainer = pl.Trainer(**config.trainer, logger=wandb_logger)
        trainer.fit(model, datamodule=datamodule)


def get_args():
    """Gets parameters from commandline."""
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="padim", help="Name of the algorithm to train/test")
    parser.add_argument("--model_config", type=Path, required=False, help="Path to a model config file")
    parser.add_argument("--sweep_config", type=Path, required=True, help="Path to sweep configuration")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model_config = get_configurable_parameters(model_name=args.model, model_config_path=args.model_config)
    hpo_config = OmegaConf.load(args.sweep_config)

    if model_config.project.seed != 0:
        seed_everything(model_config.project.seed)

    sweep = WandbSweep(model_config, hpo_config)
    sweep.run()
