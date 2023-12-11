"""Anomalib CLI."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
from jsonargparse import ActionConfigFile
from lightning.pytorch import Trainer
from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI, SaveConfigCallback
from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT, _PREDICT_OUTPUT
from rich import traceback

from anomalib.callbacks import get_callbacks, get_visualization_callbacks
from anomalib.callbacks.normalization import get_normalization_callback
from anomalib.cli.utils import CustomHelpFormatter
from anomalib.cli.utils.openvino import add_openvino_export_arguments
from anomalib.data import AnomalibDataModule, AnomalibDataset
from anomalib.engine import Engine
from anomalib.loggers import configure_logger
from anomalib.metrics.threshold import BaseThreshold
from anomalib.models import AnomalyModule
from anomalib.pipelines.benchmarking import distribute
from anomalib.pipelines.hpo import Sweep, get_hpo_parser
from anomalib.utils.config import update_config
from anomalib.utils.types import TaskType

traceback.install()
logger = logging.getLogger("anomalib.cli")


class AnomalibCLI(LightningCLI):
    """Implementation of a fully configurable CLI tool for anomalib.

    The advantage of this tool is its flexibility to configure the pipeline
    from both the CLI and a configuration file (.yaml or .json). It is even
    possible to use both the CLI and a configuration file simultaneously.
    For more details, the reader could refer to PyTorch Lightning CLI
    documentation.

    ``save_config_kwargs`` is set to ``overwrite=True`` so that the
    ``SaveConfigCallback`` overwrites the config if it already exists.
    """

    def __init__(
        self,
        save_config_callback: type[SaveConfigCallback] = SaveConfigCallback,
        save_config_kwargs: dict[str, Any] | None = None,
        trainer_class: type[Trainer] | Callable[..., Trainer] = Trainer,
        trainer_defaults: dict[str, Any] | None = None,
        seed_everything_default: bool | int = True,
        parser_kwargs: dict[str, Any] | dict[str, dict[str, Any]] | None = None,
        args: ArgsType = None,
        run: bool = True,
        auto_configure_optimizers: bool = True,
    ) -> None:
        super().__init__(
            AnomalyModule,
            AnomalibDataModule,
            save_config_callback,
            {"overwrite": True} if save_config_kwargs is None else save_config_kwargs,
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

    def init_parser(self, **kwargs) -> LightningArgumentParser:
        """Method that instantiates the argument parser."""
        kwargs.setdefault("dump_header", [f"lightning.pytorch=={pl.__version__}"])
        parser = LightningArgumentParser(formatter_class=CustomHelpFormatter, **kwargs)
        parser.add_argument(
            "-c",
            "--config",
            action=ActionConfigFile,
            help="Path to a configuration file in json or yaml format.",
        )
        return parser

    @staticmethod
    def anomalib_subcommands() -> dict[str, dict[str, str]]:
        """Return a dictionary of subcommands and their description."""
        return {
            "export": {"description": "Export the model to ONNX or OpenVINO format."},
            "benchmark": {"description": "Run benchmarking script"},
            "hpo": {"description": "Run Hyperparameter Optimization"},
            "train": {"description": "Fit the model and then call test on the trained model."},
        }

    def _add_subcommands(self, parser: LightningArgumentParser, **kwargs) -> None:
        """Initialize base subcommands and add anomalib specific on top of it."""
        # Initializes fit, validate, test, predict and tune
        super()._add_subcommands(parser, **kwargs)
        # Add  export, benchmark and hpo
        for subcommand in self.anomalib_subcommands():
            sub_parser = LightningArgumentParser(formatter_class=CustomHelpFormatter)
            self._subcommand_parsers[subcommand] = sub_parser
            self.parser._subcommands_action.add_subcommand(  # noqa: SLF001
                subcommand,
                sub_parser,
                help=self.anomalib_subcommands()[subcommand]["description"],
            )
            # add arguments to subcommand
            getattr(self, f"add_{subcommand}_arguments")(sub_parser)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Extend trainer's arguments to add engine arguments.

        .. note::
            Since ``Engine`` parameters are manually added, any change to the
            ``Engine`` class should be reflected manually.
        """
        parser.add_function_arguments(get_normalization_callback, "normalization")
        # visualization takes task from the project
        parser.add_function_arguments(get_visualization_callbacks, "visualization", skip={"task"})
        parser.add_argument("--task", type=TaskType, default=TaskType.SEGMENTATION)
        parser.add_argument("--metrics.image", type=list[str] | str | None, default=["F1Score", "AUROC"])
        parser.add_argument("--metrics.pixel", type=list[str] | str | None, default=None, required=False)
        parser.add_argument("--metrics.threshold", type=BaseThreshold, default="F1AdaptiveThreshold")
        parser.add_argument("--logging.log_graph", type=bool, help="Log the model to the logger", default=False)
        parser.link_arguments("data.init_args.image_size", "model.init_args.input_size")
        parser.link_arguments("task", "data.init_args.task")
        parser.add_argument(
            "--results_dir.path",
            type=Path,
            help="Path to save the results.",
            default=Path("./results"),
        )
        parser.add_argument("--results_dir.unique", type=bool, help="Whether to create a unique folder.", default=False)
        parser.link_arguments("results_dir.path", "trainer.default_root_dir")
        # TODO(ashwinvaidya17): Tiling should also be a category of its own
        # CVS-122659

    def add_train_arguments(self, parser: LightningArgumentParser) -> None:
        """Add train arguments to the parser."""
        parser.add_argument(
            "-c",
            "--config",
            action=ActionConfigFile,
            help="Path to a configuration file in json or yaml format.",
        )
        parser.add_lightning_class_args(self.trainer_class, "trainer")
        parser.add_lightning_class_args(AnomalyModule, "model", subclass_mode=True)
        parser.add_subclass_arguments(AnomalibDataModule, "data")
        trainer_defaults = {"trainer." + k: v for k, v in self.trainer_defaults.items() if k != "callbacks"}
        parser.set_defaults(trainer_defaults)
        added = parser.add_method_arguments(
            Engine,
            "train",
            skip={"model", "datamodule", "val_dataloaders", "test_dataloaders", "train_dataloaders"},
        )
        self._subcommand_method_arguments["train"] = added
        self.add_arguments_to_parser(parser)
        self.add_default_arguments_to_parser(parser)

    def add_export_arguments(self, parser: LightningArgumentParser) -> None:
        """Add export arguments to the parser."""
        parser.add_argument(
            "-c",
            "--config",
            action=ActionConfigFile,
            help="Path to a configuration file in json or yaml format.",
        )
        parser.add_lightning_class_args(self.trainer_class, "trainer")
        trainer_defaults = {"trainer." + k: v for k, v in self.trainer_defaults.items() if k != "callbacks"}
        parser.set_defaults(trainer_defaults)
        parser.add_lightning_class_args(AnomalyModule, "model", subclass_mode=True)
        parser.add_subclass_arguments((AnomalibDataModule, AnomalibDataset), "data")
        added = parser.add_method_arguments(
            Engine,
            "export",
            skip={"mo_args", "datamodule", "dataset", "model"},
        )
        self._subcommand_method_arguments["export"] = added
        add_openvino_export_arguments(parser)
        self.add_arguments_to_parser(parser)
        self.add_default_arguments_to_parser(parser)

    def add_hpo_arguments(self, parser: LightningArgumentParser) -> None:
        """Add hyperparameter optimization arguments."""
        parser = get_hpo_parser(parser)

    def add_benchmark_arguments(self, parser: LightningArgumentParser) -> None:
        """Add benchmark arguments to the parser."""
        parser.add_argument("--config", type=Path, help="Path to the benchmark config.", required=True)

    def before_instantiate_classes(self) -> None:
        """Modify the configuration to properly instantiate classes and sets up tiler."""
        subcommand = self.config["subcommand"]
        if subcommand in (*self.subcommands(), "train"):
            self.config[subcommand] = update_config(self.config[subcommand])

    def instantiate_classes(self) -> None:
        """Instantiate classes depending on the subcommand.

        For trainer related commands it instantiates all the model, datamodule and trainer classes.
        But for subcommands we do not want to instantiate any trainer specific classes such as datamodule, model, etc
        This is because the subcommand is responsible for instantiating and executing code based on the passed config
        """
        if self.config["subcommand"] not in self.anomalib_subcommands():
            # since all classes are instantiated, the LightningCLI also creates an unused ``Trainer`` object.
            # the minor change here is that engine is instantiated instead of trainer
            self.config_init = self.parser.instantiate_classes(self.config)
            self.datamodule = self._get(self.config_init, "data")
            self.model = self._get(self.config_init, "model")
            self._add_configure_optimizers_method_to_model(self.subcommand)
            self.engine = self.instantiate_engine()
        else:
            self.config_init = self.parser.instantiate_classes(self.config)
            subcommand = self.config["subcommand"]
            if subcommand in ("train", "export"):
                self.engine = self.instantiate_engine()
            if "model" in self.config_init[subcommand]:
                self.model = self._get(self.config_init, "model")
            if "data" in self.config_init[subcommand]:
                self.datamodule = self._get(self.config_init, "data")

    def instantiate_engine(self) -> Engine:
        """Instantiate the engine.

        .. note::
            Most of the code in this method is taken from ``LightningCLI``'s
            ``instantiate_trainer`` method. Refer to that method for more
            details.
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

    def _run_subcommand(self, subcommand: str) -> None:
        """Run subcommand depending on the subcommand.

        This overrides the original ``_run_subcommand`` to run the ``Engine``
        method rather than the ``Train`` method.
        """
        if self.config["subcommand"] in (*self.subcommands(), "train", "export"):
            fn = getattr(self.engine, subcommand)
            fn_kwargs = self._prepare_subcommand_kwargs(subcommand)
            fn(**fn_kwargs)
        else:
            self.config_init = self.parser.instantiate_classes(self.config)
            getattr(self, f"{subcommand}")()

    @property
    def fit(self) -> Callable[..., None]:
        """Fit the model using engine's fit method."""
        return self.engine.fit

    @property
    def validate(self) -> Callable[..., _EVALUATE_OUTPUT | None]:
        """Validate the model using engine's validate method."""
        return self.engine.validate

    @property
    def test(self) -> Callable[..., _EVALUATE_OUTPUT]:
        """Test the model using engine's test method."""
        return self.engine.test

    @property
    def predict(self) -> Callable[..., _PREDICT_OUTPUT | None]:
        """Predict using engine's predict method."""
        return self.engine.predict

    @property
    def train(self) -> Callable[..., _EVALUATE_OUTPUT]:
        """Train the model using engine's train method."""
        return self.engine.train

    @property
    def export(self) -> Callable[..., None]:
        """Export the model using engine's export method."""
        return self.engine.export

    def hpo(self) -> None:
        """Run hpo subcommand."""
        config = self.config["hpo"]
        sweep = Sweep(
            project=config.project,
            sweep_config=config.sweep_config,
            backend=config.backend,
            entity=config.entity,
        )
        sweep.run()

    def benchmark(self) -> None:
        """Run benchmark subcommand."""
        config = self.config["benchmark"]
        distribute(config.config)


def main() -> None:
    """Trainer via Anomalib CLI."""
    configure_logger()
    AnomalibCLI()


if __name__ == "__main__":
    main()
