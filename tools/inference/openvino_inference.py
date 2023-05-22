"""Anomalib Inferencer Script.

This script performs inference by reading a model config file from
command line, and show the visualization results.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from anomalib.utils.cli.subcommands.infer import OpenVINOInference, get_openvino_parser

if __name__ == "__main__":
    args = get_openvino_parser().parse_args()
    OpenVINOInference(
        weights=args.weights,
        metadata=args.metadata,
        input=args.input,
        output=args.output,
        task=args.task,
        device=args.device,
        visualization_mode=args.visualization_mode,
        show=args.show,
    ).run()
