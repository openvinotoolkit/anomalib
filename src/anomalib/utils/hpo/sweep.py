"""Utilities to run hyperparameter optimization."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser
from enum import Enum
from pathlib import Path

from lightning.pytorch import seed_everything
from lightning.pytorch.cli import LightningArgumentParser
from omegaconf import OmegaConf

from anomalib.config import get_configurable_parameters

from .runners import CometSweep, WandbSweep


class HPOBackend(str, Enum):
    """HPO backend choices."""

    COMET = "comet"
    WANDB = "wandb"

    def __str__(self) -> str:
        """Return the string representation of the enum value."""
        return self.value


def get_hpo_parser(
    parser: ArgumentParser | LightningArgumentParser | None = None,
) -> ArgumentParser | LightningArgumentParser:
    """Gets the HPO parser."""
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="padim",
        help="Name of the algorithm to train/test",
        required=False,
    )
    parser.add_argument("--model_config", type=Path, required=False, help="Path to a model config file")
    parser.add_argument("--sweep_config", type=Path, required=True, help="Path to sweep configuration")
    parser.add_argument(
        "--backend",
        type=HPOBackend,
        default=HPOBackend.WANDB,
        help="HPO backend to use",
        required=False,
    )
    parser.add_argument(
        "--entity",
        type=str,
        required=False,
        help="Username or workspace where you want to send your runs to. If not set, the default workspace is used.",
    )

    return parser


class Sweep:
    """HPO class to run hyperparameter optimization.

    Args:
        model (str | None): Name of the algorithm to train/test. If not provided, the model name is read from the model
         config.
        model_config (Path | str | None): Path to a model config file. If not provided, the model is loaded
            based on the model name.
        sweep_config (Path | str): Path to sweep configuration. The configuration depends on the type
            of backend.
        backend (HPOBackend): HPO backend to use. Defaults to ```Backend.COMET```.
        entity (str | None): Username or workspace where you want to send your runs to. If not set, the default
            workspace is used.
    """

    def __init__(
        self,
        model: str | None,
        model_config: Path | str | None,
        sweep_config: Path | str,
        backend: HPOBackend = HPOBackend.COMET,
        entity: str = "",
    ) -> None:
        if model is None and model_config is None:
            msg = "Either model or model_config must be provided."
            raise ValueError(msg)

        self.model_config = get_configurable_parameters(model_name=model, config_path=model_config)
        self.sweep_config = OmegaConf.load(sweep_config)
        self.entity = entity

        self.runner = self.get_runner(backend)

    def get_runner(self, backend: HPOBackend) -> CometSweep | WandbSweep:
        """Gets the runner for the specified backend."""
        runner: CometSweep | WandbSweep
        if backend == HPOBackend.COMET:
            runner = CometSweep(self.model_config, self.sweep_config, self.entity)
        elif backend == HPOBackend.WANDB:
            runner = WandbSweep(self.model_config, self.sweep_config, self.entity)
        else:
            msg = f"Unknown backend {backend}"
            raise ValueError(msg)
        return runner

    def run(self) -> None:
        """Runs the sweep."""
        if self.model_config.get("seed_everything") is not None:
            seed_everything(self.model_config.seed_everything)

        self.runner.run()
