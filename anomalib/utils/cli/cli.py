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
from anomalib.deploy.export import _export_to_onnx, _export_to_openvino
from anomalib.models import get_model
from anomalib.pre_processing.tiler import TilerDecorator
from anomalib.utils.benchmark import distribute
from anomalib.utils.callbacks import (
    ImageVisualizerCallback,
    MetricsConfigurationCallback,
    PostProcessingConfigurationCallback,
    get_callbacks_dict,
)
from anomalib.utils.hpo import Sweep, get_hpo_parser
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
        """Instantiate classes depending on the subcommand.

        For trainer related commands it instantiates all the model, datamodule and trainer classes.
        But for export and hpo we do not want to instantiate any classes.
        """
        if self.config["subcommand"] not in self.anomalib_subcommands():
            super().instantiate_classes()

    def parse_arguments(self, parser: LightningArgumentParser) -> None:
        """Parse arguments depending on the subcommand.

        For export and hpo we do not want to check parameters such as model, datamodule, trainer, etc.
        """
        if len(sys.argv) > 1 and sys.argv[1] in self.anomalib_subcommands():
            # this ensures that lightning parameters are not checked in the parser
            parser._choices.clear()  # pylint: disable=protected-access
        super().parse_arguments(parser)

    def _run_subcommand(self, subcommand: str) -> None:
        """Run subcommand depending on the subcommand."""
        if self.config["subcommand"] not in self.anomalib_subcommands():
            super()._run_subcommand(subcommand)
        else:
            getattr(self, f"run_{subcommand}")()

    def run_export(self) -> None:
        """Run export subcommand."""
        config = self.config["export"]
        if config.export_mode == "onnx":
            # load model
            config = config["onnx"]
            model_config = get_configurable_parameters(config_path=config.model_config)
            model = get_model(model_config)
            model.load_state_dict(torch.load(config.weights)["state_dict"])
            export_path = config.weights.parent if config.export_path is None else config.export_path
            onnx_path = _export_to_onnx(model, model_config.model.init_args.input_size, export_path)
            print(f"Model exported to {onnx_path}")
        elif config.export_mode == "openvino":
            config = config["openvino"]
            mo_parser = get_common_cli_parser()
            mo_config = {"framework": "onnx"}
            # Get default parameters from the openvino subcommand of the export sub parser of self.parser
            for key, value in config.mo.items():
                if mo_parser.get_default(key) != value:
                    mo_config[key] = value
            _export_to_openvino(export_path=config.export_path, input_model=config.input_model, **mo_config)
            print(f"Model exported to {config.export_path}")

    def run_hpo(self) -> None:
        """Run hpo subcommand."""
        config = self.config["hpo"]
        sweep = Sweep(
            model=config.model,
            model_config=config.model_config,
            sweep_config=config.sweep_config,
            backend=config.backend,
        )
        sweep.run()

    def run_benchmark(self) -> None:
        """Run benchmark subcommand."""
        config = self.config["benchmark"]
        distribute(config.config)

    @staticmethod
    def anomalib_subcommands() -> Dict[str, Dict[str, Any]]:
        """Returns a dictionary of subcommands and their description."""
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
        for subcommand in self.anomalib_subcommands():
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
        openvino_parser.add_argument(
            "--export_path",
            type=str,
            help="Path to save the exported model.",
        )
        openvino_parser.add_argument("--input_model", type=Path, help="Path to the torch model weights.", required=True)
        group = openvino_parser.add_argument_group("OpenVINO Model Optimizer arguments (optional)")
        mo_parser = get_common_cli_parser()

        # remove redundant keys from mo keys
        for arg in mo_parser._actions:
            if arg.dest in ("help", "input_model", "output_dir"):
                continue
            group.add_argument(f"--mo.{arg.dest}", type=arg.type, default=arg.default, help=arg.help)

    def add_hpo_arguments(self, parser: LightningArgumentParser) -> None:
        """Add hyperparameter optimization arguments."""
        parser = get_hpo_parser(parser)

    def add_benchmark_arguments(self, parser: LightningArgumentParser) -> None:
        """Adds benchmark arguments to the parser."""
        parser.add_argument("--config", type=Path, help="Path to the benchmark config.", required=True)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add custom arguments to the parser."""
        # TODO: https://github.com/openvinotoolkit/anomalib/issues/20
        parser.add_class_arguments(ImageVisualizerCallback, "visualization")
        parser.add_class_arguments(TilerDecorator, "tiling")  # TODO tiler should be added to each model.
        parser.add_class_arguments(MetricsConfigurationCallback, "metrics")
        parser.add_class_arguments(PostProcessingConfigurationCallback, "post_processing")
        parser.link_arguments("data.init_args.task", "visualization.task")
        parser.link_arguments("data.init_args.image_size", "model.init_args.input_size")
        parser.add_argument("--results_dir.path", type=Path, help="Path to save the results.")
        parser.add_argument("--results_dir.unique", type=bool, help="Whether to create a unique folder.", default=False)

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
