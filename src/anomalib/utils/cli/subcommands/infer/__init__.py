"""Utilities related to inference."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from jsonargparse import ActionConfigFile, ArgumentParser, Namespace

from .gradio_inference import GradioInference, get_gradio_parser
from .lightning_inference import LightningInference, get_lightning_parser
from .openvino_inference import OpenVINOInference, get_openvino_parser
from .torch_inference import TorchInference, get_torch_parser

__all__ = [
    "add_inference_parser",
    "get_gradio_parser",
    "get_lightning_parser",
    "get_openvino_parser",
    "get_torch_parser",
    "run_inference",
    "GradioInference",
    "LightningInference",
    "OpenVINOInference",
    "TorchInference",
]


def add_inference_parser(parser: ArgumentParser):
    """Method that instantiates the argument parser."""
    sub_parser = ArgumentParser()
    sub_parser.add_argument(
        "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
    )
    parser._subcommands_action.add_subcommand(
        "infer", sub_parser, help="Call inference on the model using either Torch, OpenVINO, Gradio or Lightning models"
    )
    subparsers = sub_parser.add_subcommands(dest="backend", help="Inference backend")
    subparsers.add_subcommand("openvino", get_openvino_parser(), help="Inference using OpenVINO model")
    subparsers.add_subcommand("torch", get_torch_parser(), help="Inference using PyTorch model")
    subparsers.add_subcommand("lightning", get_lightning_parser(), help="Inference using Lightning model")
    subparsers.add_subcommand(
        "gradio", get_gradio_parser(), help="Gradio inference using either Torch, OpenVINO or Lightning model"
    )


def run_inference(config: Namespace):
    """Run infer subcommand."""
    backend = config.backend
    config = config[backend]
    match backend:
        case "gradio":
            GradioInference(**config).run()
        case "torch":
            TorchInference(**config).run()
        case "openvino":
            OpenVINOInference(**config).run()
        case "lightning":
            LightningInference(**config).run()
        case _:
            raise ValueError(f"Unknown inference backend {backend}")
