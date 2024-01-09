"""Utilities to run hyperparameter optimization."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from pathlib import Path

from lightning.pytorch.cli import LightningArgumentParser
from omegaconf import OmegaConf

from .runners import CometSweep, WandbSweep


class HPOBackend(str, Enum):
    """HPO backend choices."""

    COMET = "comet"
    WANDB = "wandb"

    def __str__(self) -> str:
        """Return the string representation of the enum value."""
        return self.value


def get_hpo_parser(
    parser: LightningArgumentParser | None = None,
) -> LightningArgumentParser:
    """Get the HPO parser."""
    if parser is None:
        parser = LightningArgumentParser()
    parser.add_argument("--sweep_config", type=Path, required=True, help="Path to sweep configuration")
    parser.add_argument("--project", type=str, default="AnomalibHPOSweep", help="Name of the project")
    parser.add_argument(
        "--backend",
        type=HPOBackend,
        default=HPOBackend.COMET,
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
        project (str): Name of the project.
        sweep_config (Path | str): Path to sweep configuration. The configuration depends on the type
            of backend.
        backend (HPOBackend): HPO backend to use. Defaults to ```Backend.COMET```.
        entity (str | None): Username or workspace where you want to send your runs to. If not set, the default
            workspace is used.
    """

    def __init__(
        self,
        project: str,
        sweep_config: Path | str,
        backend: HPOBackend = HPOBackend.COMET,
        entity: str = "",
    ) -> None:
        self.sweep_config = OmegaConf.load(sweep_config)
        self.entity = entity
        self.project = project

        self.runner = self.get_runner(backend)

    def get_runner(self, backend: HPOBackend) -> CometSweep | WandbSweep:
        """Get the runner for the specified backend."""
        runner: CometSweep | WandbSweep
        if backend == HPOBackend.COMET:
            runner = CometSweep(self.project, self.sweep_config, self.entity)
        elif backend == HPOBackend.WANDB:
            runner = WandbSweep(self.project, self.sweep_config, self.entity)
        else:
            msg = f"Unknown backend {backend}"
            raise ValueError(msg)
        return runner

    def run(self) -> None:
        """Run the sweep."""
        self.runner.run()
