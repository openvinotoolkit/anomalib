"""Anomalib Gradio Script.

This script provide a gradio web interface
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from anomalib.utils.cli.subcommands.infer import GradioInference, get_gradio_parser

if __name__ == "__main__":
    args = get_gradio_parser().parse_args()
    GradioInference(weights=args.weights, metadata=args.metadata, share=args.share).run()
