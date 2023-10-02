"""Anomalib CLI."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union

from jsonargparse import ArgumentParser
from lightning.pytorch import Trainer
from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI, SaveConfigCallback
from openvino.tools.mo.utils.cli_parser import get_onnx_cli_parser

from anomalib.config.config import update_config
from anomalib.data import AnomalibDataModule, TaskType
from anomalib.engine import Engine
from anomalib.models import AnomalyModule
from anomalib.utils.callbacks import get_callbacks, get_visualization_callbacks
from anomalib.utils.callbacks.normalization import get_normalization_callback
from anomalib.utils.loggers import configure_logger
from anomalib.utils.metrics.threshold import BaseThreshold

logger = logging.getLogger("anomalib.cli")


class AnomalibCLI(LightningCLI):
    """Implementation of a fully configurable CLI tool for anomalib.

    The advantage of this tool is its flexibility to configure the pipeline
    from both the CLI and a configuration file (.yaml or .json). It is even
    possible to use both the CLI and a configuration file simultaneously.
    For more details, the reader could refer to PyTorch Lightning CLI documentation.
    """

    def __init__(
        self,
        save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_kwargs: Optional[Dict[str, Any]] = None,
        trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = Trainer,
        trainer_defaults: Optional[Dict[str, Any]] = None,
        seed_everything_default: Union[bool, int] = True,
        parser_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
        args: ArgsType = None,
        run: bool = True,
        auto_configure_optimizers: bool = True,
    ) -> None:
        super().__init__(
            AnomalyModule,
            AnomalibDataModule,
            save_config_callback,
            save_config_kwargs,
            trainer_class,
            trainer_defaults,
            seed_everything_default,
            parser_kwargs,
            subclass_mode_model=True,
            subclass_mode_data=True,
            args=args,
            run=run,
            auto_configure_optimizers=auto_configure_optimizers,
        )
        self.engine: Engine

    @staticmethod
    def anomalib_subcommands() -> dict[str, dict[str, Any]]:
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

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Extend trainer's arguments to add engine arguments.

        Note:
            Since ``Engine`` parameters are manually added, any change to the ``Engine`` class should be reflected
            manually.
        """
        parser.add_function_arguments(get_normalization_callback, "normalization")
        # visualization takes task from the project
        parser.add_function_arguments(get_visualization_callbacks, "visualization", skip={"task"})
        parser.add_argument("task", type=TaskType, default=TaskType.SEGMENTATION)
        parser.add_argument("metrics.image", type=list[str] | str | None, default=["F1Score", "AUROC"])
        parser.add_argument("metrics.pixel", type=list[str] | str | None, default=["F1Score", "AUROC"])
        parser.add_argument("metrics.threshold", type=BaseThreshold, default="F1AdaptiveThreshold")
        parser.add_argument("--logging.log_graph", type=bool, help="Log the model to the logger", default=False)
        parser.link_arguments("data.init_args.image_size", "model.init_args.input_size")
        parser.link_arguments("task", "data.init_args.task")
        parser.add_argument("--results_dir.path", type=Path, help="Path to save the results.")
        parser.add_argument("--results_dir.unique", type=bool, help="Whether to create a unique folder.", default=False)
        parser.link_arguments("results_dir.path", "trainer.default_root_dir")
        # TODO tiling should also be a category of its own

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
        mo_parser = get_onnx_cli_parser()

        # remove redundant keys from mo keys
        for arg in mo_parser._actions:
            if arg.dest in ("help", "input_model", "output_dir"):
                continue
            group.add_argument(f"--mo.{arg.dest}", type=arg.type, default=arg.default, help=arg.help)

    def add_hpo_arguments(self, parser: LightningArgumentParser) -> None:
        """Add hyperparameter optimization arguments."""
        # parser = get_hpo_parser(parser)

    def add_benchmark_arguments(self, parser: LightningArgumentParser) -> None:
        """Adds benchmark arguments to the parser."""
        parser.add_argument("--config", type=Path, help="Path to the benchmark config.", required=True)

    def parse_arguments(self, parser: LightningArgumentParser, args: ArgsType) -> None:
        """Parse arguments depending on the subcommand.

        For export and hpo we do not want to check parameters such as model, datamodule, trainer, etc.
        """
        super().parse_arguments(parser, args)

    def before_instantiate_classes(self) -> None:
        """Modify the configuration to properly instantiate classes and sets up tiler."""
        subcommand = self.config["subcommand"]
        if subcommand not in self.anomalib_subcommands():
            self.config[subcommand] = update_config(self.config[subcommand])

    def instantiate_classes(self) -> None:
        """Instantiate classes depending on the subcommand.

        For trainer related commands it instantiates all the model, datamodule and trainer classes.
        But for export and hpo we do not want to instantiate any classes.
        """
        if self.config["subcommand"] not in self.anomalib_subcommands():
            # since all classes are instantiated, the LightningCLI also creates an unused ``Trainer`` object.
            self.config_init = self.parser.instantiate_classes(self.config)
            self.datamodule = self._get(self.config_init, "data")
            self.model = self._get(self.config_init, "model")
            self._add_configure_optimizers_method_to_model(self.subcommand)
            self.engine = self.instantiate_engine()

    def instantiate_engine(self) -> Engine:
        """Instantiate the engine.

        Note:
            Most of the code in this method is taken from LightningCLI's ``instantiate_trainer`` method.
            Refer to that method for more details.
        """
        extra_callbacks = [self._get(self.config_init, c) for c in self._parser(self.subcommand).callback_keys]
        engine_args = {
            "normalization": self._get(self.config_init, "normalization.normalization_method"),
            "threshold": self._get(self.config_init, "metrics.threshold"),
            "task": self._get(self.config_init, "task"),
            "image_metrics": self._get(self.config_init, "metrics.image"),
            "pixel_metrics": self._get(self.config_init, "metrics.pixel"),
            "visualization": self._get(self.config_init, "visualization"),
        }
        trainer_config = {**self._get(self.config_init, "trainer", default={}), **engine_args}
        key = "callbacks"
        if key in trainer_config:
            if trainer_config[key] is None:
                trainer_config[key] = []
            elif not isinstance(trainer_config[key], list):
                trainer_config[key] = [trainer_config[key]]
            trainer_config[key].extend(extra_callbacks)
            if key in self.trainer_defaults:
                value = self.trainer_defaults[key]
                trainer_config[key] += value if isinstance(value, list) else [value]
            if self.save_config_callback and not trainer_config.get("fast_dev_run", False):
                config_callback = self.save_config_callback(
                    self._parser(self.subcommand),
                    self.config.get(str(self.subcommand), self.config),
                    **self.save_config_kwargs,
                )
                trainer_config[key].append(config_callback)
        trainer_config[key].extend(get_callbacks(self.config[self.subcommand]))
        return Engine(**trainer_config)

    @property
    def fit(self):
        return self.engine.fit

    @property
    def test(self):
        return self.engine.test

    @property
    def validate(self):
        return self.engine.validate

    @property
    def predict(self):
        return self.engine.predict

    def _run_subcommand(self, subcommand: str) -> None:
        """Run subcommand depending on the subcommand.

        This overrides the original ``_run_subcommand`` to run the ``Engine`` method rather than the ``Train`` method
        """
        if self.config["subcommand"] not in self.anomalib_subcommands():
            fn = getattr(self.engine, subcommand)
            fn_kwargs = self._prepare_subcommand_kwargs(subcommand)
            fn(**fn_kwargs)
        else:
            getattr(self, f"run_{subcommand}")()


def main() -> None:
    """Trainer via Anomalib CLI."""
    configure_logger()
    AnomalibCLI()


if __name__ == "__main__":
    main()
