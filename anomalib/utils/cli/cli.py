"""Anomalib CLI."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union

import torch
from jsonargparse import ArgumentParser
from openvino.tools.mo.utils.cli_parser import get_common_cli_parser
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.utilities.cli import (
    LightningArgumentParser,
    LightningCLI,
    SaveConfigCallback,
)

from anomalib.config import get_configurable_parameters, update_config
from anomalib.deploy.export import _export_to_onnx
from anomalib.models import get_model
from anomalib.pre_processing.tiler import TilerDecorator
from anomalib.utils.callbacks import (
    ImageVisualizerCallback,
    MetricsConfigurationCallback,
    PostProcessingConfigurationCallback,
    get_callbacks_dict,
)
from anomalib.utils.loggers import configure_logger

logger = logging.getLogger("anomalib.cli")


class AnomalibCLI(LightningCLI):
    """Implementation of a fully configurable CLI tool for anomalib.

    The advantage of this tool is its flexibility to configure the pipeline
    from both the CLI and a configuration file (.yaml or .json). It is even
    possible to use both the CLI and a configuration file simultaneously.
    For more details, the reader could refer to PyTorch Lightning CLI documentation.
    """

    def __init__(  # pylint: disable=too-many-function-args
        self,
        model_class: Optional[Union[Type[LightningModule], Callable[..., LightningModule]]] = None,
        datamodule_class: Optional[Union[Type[LightningDataModule], Callable[..., LightningDataModule]]] = None,
        save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_filename: str = "config.yaml",
        save_config_overwrite: bool = False,
        save_config_multifile: bool = False,
        trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = Trainer,
        trainer_defaults: Optional[Dict[str, Any]] = None,
        seed_everything_default: Optional[int] = None,
        description: str = "Anomalib trainer command line tool",
        env_prefix: str = "Anomalib",
        env_parse: bool = False,
        parser_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
        run: bool = True,
        auto_registry: bool = True,
    ) -> None:
        super().__init__(
            model_class,
            datamodule_class,
            save_config_callback,
            save_config_filename,
            save_config_overwrite,
            save_config_multifile,
            trainer_class,
            trainer_defaults,
            seed_everything_default,
            description,
            env_prefix,
            env_parse,
            parser_kwargs,
            subclass_mode_model,
            subclass_mode_data,
            run,
            auto_registry,
        )

    def instantiate_classes(self) -> None:
        if self.config["subcommand"] not in self.anomalib_subcommands():
            return super().instantiate_classes()

    def parse_arguments(self, parser: LightningArgumentParser) -> None:
        """Parse arguments depending on the subcommand."""
        if len(sys.argv) > 1 and sys.argv[1] in self.anomalib_subcommands().keys():
            parser._choices.clear()  # this ensures that lightning parameters are not checked in the parser
            arguments = super().parse_arguments(parser)
        else:
            arguments = super().parse_arguments(parser)
        return arguments

    def _run_subcommand(self, subcommand: str) -> None:
        if self.config["subcommand"] not in self.anomalib_subcommands():
            return super()._run_subcommand(subcommand)
        else:
            getattr(self, f"run_{subcommand}")()

    def run_export(self) -> None:
        """Run export subcommand."""
        config = self.config["export"]
        # load model
        if config.export_mode == "onnx":
            config = config["onnx"]
            model_config = get_configurable_parameters(config_path=config.model_config)
            model = get_model(model_config)
            model.load_state_dict(torch.load(config.weights)["state_dict"])
            export_path = config.weights.parent if config.export_path is None else config.export_path
            onnx_path = _export_to_onnx(model, model_config.model.init_args.input_size, export_path)
            print(f"Model exported to {onnx_path}")

    def run_hpo(self) -> None:
        raise NotImplementedError("Hyperparameter Optimization is not implemented yet.")

    def run_benchmark(self) -> None:
        raise NotImplementedError("Benchmarking is not implemented yet.")

    @staticmethod
    def anomalib_subcommands() -> Dict[str, Dict[str, Any]]:
        return {
            "export": {"description": "Export the model to ONNX or OpenVINO format."},
            "benchmark": {"description": "Run benchmarking script"},
            "hpo": {"description": "Run Hyperparameter Optimization"},
        }

    def _add_subcommands(self, parser: LightningArgumentParser, **kwargs: Any) -> None:
        """Setup base subcommands and add anomalib specific on top of it."""
        # Initializes fit, validate, test, predict and tune
        super()._add_subcommands(parser, **kwargs)
        # Add  export, benchmark and hpo
        for subcommand in self.anomalib_subcommands().keys():
            sub_parser = ArgumentParser()
            self.parser._subcommands_action.add_subcommand(
                subcommand, sub_parser, help=self.anomalib_subcommands()[subcommand]["description"]
            )
            # add arguments to subcommand
            getattr(self, f"add_{subcommand}_arguments")(sub_parser)

    def add_export_arguments(self, parser: LightningArgumentParser) -> None:
        """Adds export arguments to the parser."""
        subcommands = parser.add_subcommands(dest="export_mode", help="Export mode. ONNX or OpenVINO")
        # Add export mode sub parsers
        onnx_parser = LightningArgumentParser(description="Export to ONNX format")
        subcommands.add_subcommand("onnx", onnx_parser)
        onnx_parser.add_argument("--weights", type=Path, help="Path to the torch model weights.", required=True)
        onnx_parser.add_argument("--model_config", type=Path, help="Path to the model config.", required=True)
        onnx_parser.add_argument("--export_path", type=Path, help="Path to save the exported model.")

        openvino_parser = LightningArgumentParser(description="Export to OpenVINO format")
        subcommands.add_subcommand("openvino", openvino_parser)
        # parser.add_argument(
        #     "--output_path",
        #     type=str,
        #     default=".",
        #     help="Path to save the exported model.",
        # )
        group = openvino_parser.add_argument_group("OpenVINO Model Optimizer arguments (optional)")
        mo_parser = get_common_cli_parser()
        for arg in mo_parser._actions:
            if arg.dest in ("help", "output_dir"):
                continue
            group.add_argument(f"--mo.{arg.dest}", type=arg.type, default=arg.default, help=arg.help)

    def add_hpo_arguments(self, parser: LightningArgumentParser) -> None:
        """Add hyperparameter optimization arguments."""
        parser.add_argument(
            "--backend",
            type=str,
            default="wandb",
            help="Select which backend to use for running HPO.",
            choices=["wandb", "comet"],
        )
        pass

    def add_benchmark_arguments(self, parser: LightningArgumentParser) -> None:
        """Adds benchmark arguments to the parser."""
        pass

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add custom arguments to the parser."""
        # TODO: https://github.com/openvinotoolkit/anomalib/issues/20
        parser.add_class_arguments(ImageVisualizerCallback, "visualization")
        parser.add_class_arguments(TilerDecorator, "tiling")  # TODO tiler should be added to each model.
        parser.add_class_arguments(MetricsConfigurationCallback, "metrics")
        parser.add_class_arguments(PostProcessingConfigurationCallback, "post_processing")
        parser.link_arguments("data.init_args.task", "visualization.task")
        parser.link_arguments("data.init_args.image_size", "model.init_args.input_size")
        parser.add_argument("results_dir.path", type=Path, help="Path to save the results.")
        parser.add_argument("results_dir.unique", type=bool, help="Whether to create a unique folder.", default=False)

        # parser.set_defaults("visualization.image_save_path",)

    def before_instantiate_classes(self) -> None:
        """Modify the configuration to properly instantiate classes and sets up tiler."""
        subcommand = self.config["subcommand"]
        if subcommand not in self.anomalib_subcommands():
            self.config[subcommand] = update_config(self.config[subcommand])
            TilerDecorator(**vars(self.config[subcommand].tiling))
            self.config[subcommand].trainer.callbacks = get_callbacks_dict(self.config[subcommand])


def main() -> None:
    """Trainer via Anomalib CLI."""
    configure_logger()
    AnomalibCLI()


if __name__ == "__main__":
    main()
