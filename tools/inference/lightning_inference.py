"""Inference Entrypoint script."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.utils.cli.subcommands.infer import LightningInference, get_lightning_parser

if __name__ == "__main__":
    args = get_lightning_parser().parse_args()
    LightningInference(
        config=args.config,
        weights=args.weights,
        input=args.input,
        output=args.output,
        visualization_mode=args.visualization_mode,
        show=args.show,
    ).run()
