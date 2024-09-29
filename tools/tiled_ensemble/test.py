"""Run tiled ensemble training."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser
from pathlib import Path

from anomalib.pipelines.tiled_ensemble import TestTiledEnsemble


def get_parser() -> ArgumentParser:
    """Create a new parser if none is provided."""
    parser = ArgumentParser()
    parser.add_argument("--config", type=str | Path, help="Configuration file path.", required=True)
    parser.add_argument("--root", type=str | Path, help="Weights file path.", required=True)

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    print("Running tiled ensemble test pipeline.")
    # pass the path to root dir with checkpoints
    test_pipeline = TestTiledEnsemble(args.root)
    test_pipeline.run(args.config)
