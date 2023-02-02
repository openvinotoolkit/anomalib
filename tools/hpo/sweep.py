"""Run hpo sweep."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from anomalib.config import get_configurable_parameters
from anomalib.utils.hpo import CometSweep, WandbSweep


def get_args():
    """Gets parameters from commandline."""
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="padim", help="Name of the algorithm to train/test")
    parser.add_argument("--model_config", type=Path, required=False, help="Path to a model config file")
    parser.add_argument("--sweep_config", type=Path, required=True, help="Path to sweep configuration")
    parser.add_argument(
        "--entity",
        type=str,
        required=False,
        help="Username or workspace where you want to send your runs to. If not set, the default workspace is used.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model_config = get_configurable_parameters(model_name=args.model, config_path=args.model_config)
    hpo_config = OmegaConf.load(args.sweep_config)

    if model_config.project.get("seed") is not None:
        seed_everything(model_config.project.seed)

    # check hpo config structure to see whether it adheres to comet or wandb format
    sweep: CometSweep | WandbSweep
    if "spec" in hpo_config.keys():
        sweep = CometSweep(model_config, hpo_config, entity=args.entity)
    else:
        sweep = WandbSweep(model_config, hpo_config, entity=args.entity)
    sweep.run()
