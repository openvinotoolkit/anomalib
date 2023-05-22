"""Anomalib Torch Inferencer Script.

This script performs torch inference by reading model weights
from command line, and show the visualization results.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.utils.cli.subcommands.infer import TorchInference, get_torch_parser

if __name__ == "__main__":
    args = get_torch_parser().parse_args()
    TorchInference(
        weights=args.weights,
        input=args.input,
        output=args.output,
        device=args.device,
        task=args.task,
        visualization_mode=args.visualization_mode,
        show=args.show,
    ).run()
