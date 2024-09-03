"""Anomalib CLI."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from types import MethodType
from typing import Any

from jsonargparse import ActionConfigFile, ArgumentParser, Namespace
from jsonargparse._actions import _ActionSubCommands
from rich import traceback

from anomalib import TaskType, __version__
from anomalib.cli.pipelines import PIPELINE_REGISTRY, pipeline_subcommands, run_pipeline
from anomalib.cli.utils.help_formatter import CustomHelpFormatter, get_short_docstring
from anomalib.cli.utils.openvino import add_openvino_export_arguments
from anomalib.loggers import configure_logger

traceback.install()
logger = logging.getLogger("anomalib.cli")

_LIGHTNING_AVAILABLE = True
try:
    from lightning.pytorch import Trainer
    from torch.utils.data import DataLoader, Dataset

    from anomalib.data import AnomalibDataModule
    from anomalib.engine import Engine
    from anomalib.metrics.threshold import Threshold
    from anomalib.models import AnomalyModule
    from anomalib.utils.config import update_config

except ImportError:
    _LIGHTNING_AVAILABLE = False


class AnomalibCLI:
    """Implementation of a fully configurable CLI tool for anomalib.

    The advantage of this tool is its flexibility to configure the pipeline
    from both the CLI and a configuration file (.yaml or .json). It is even
    possible to use both the CLI and a configuration file simultaneously.
    For more details, the reader could refer to PyTorch Lightning CLI
    documentation.

    ``save_config_kwargs`` is set to ``overwrite=True`` so that the
    ``SaveConfigCallback`` overwrites the config if it already exists.
    """

    def __init__(self, args: Sequence[str] | None = None, run: bool = True) -> None:
        self.parser = self.init_parser()
        self.subcommand_parsers: dict[str, ArgumentParser] = {}
        self.subcommand_method_arguments: dict[str, list[str]] = {}
        self.add_subcommands()
        self.config = self.parser.parse_args(args=args)
        self.subcommand = self.config["subcommand"]
        if _LIGHTNING_AVAILABLE:
            self.before_instantiate_classes()
            self.instantiate_classes()
        if run:
            self._run_subcommand()

    @staticmethod
    def init_parser(**kwargs) -> ArgumentParser:
        """Method that instantiates the argument parser."""
        kwargs.setdefault("dump_header", [f"anomalib=={__version__}"])
        parser = ArgumentParser(formatter_class=CustomHelpFormatter, **kwargs)
        parser.add_argument(
            "-c",
            "--config",
            action=ActionConfigFile,
            help="Path to a configuration file in json or yaml format.",
        )
        return parser

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        """Skip predict subcommand as it is added later."""
        return {
            "fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule"},
            "validate": {"model", "dataloaders", "datamodule"},
            "test": {"model", "dataloaders", "datamodule"},
        }

    @staticmethod
    def anomalib_subcommands() -> dict[str, dict[str, str]]:
        """Return a dictionary of subcommands and their description."""
        return {
            "train": {"description": "Fit the model and then call test on the trained model."},
            "predict": {"description": "Run inference on a model."},
            "export": {"description": "Export the model to ONNX or OpenVINO format."},
        }

    def add_subcommands(self, **kwargs) -> None:
        """Initialize base subcommands and add anomalib specific on top of it."""
        parser_subcommands = self.parser.add_subcommands()

        # Extra subcommand: install
        self._set_install_subcommand(parser_subcommands)

        if not _LIGHTNING_AVAILABLE:
            # If environment is not configured to use pl, do not add a subcommand for Engine.
            return

        # Add Trainer subcommands
        for subcommand in self.subcommands():
            sub_parser = self.init_parser(**kwargs)

            fn = getattr(Trainer, subcommand)
            # extract the first line description in the docstring for the subcommand help message
            description = get_short_docstring(fn)
            subparser_kwargs = kwargs.get(subcommand, {})
            subparser_kwargs.setdefault("description", description)

            self.subcommand_parsers[subcommand] = sub_parser
            parser_subcommands.add_subcommand(subcommand, sub_parser, help=description)
            self.add_trainer_arguments(sub_parser, subcommand)

        # Add anomalib subcommands
        for subcommand in self.anomalib_subcommands():
            sub_parser = self.init_parser(**kwargs)

            self.subcommand_parsers[subcommand] = sub_parser
            parser_subcommands.add_subcommand(
                subcommand,
                sub_parser,
                help=self.anomalib_subcommands()[subcommand]["description"],
            )
            # add arguments to subcommand
            getattr(self, f"add_{subcommand}_arguments")(sub_parser)

        # Add pipeline subcommands
        if PIPELINE_REGISTRY is not None:
            for subcommand, value in pipeline_subcommands().items():
                sub_parser = PIPELINE_REGISTRY[subcommand].get_parser()
                self.subcommand_parsers[subcommand] = sub_parser
                parser_subcommands.add_subcommand(subcommand, sub_parser, help=value["description"])

    @staticmethod
    def add_arguments_to_parser(parser: ArgumentParser) -> None:
        """Extend trainer's arguments to add engine arguments.

        .. note::
            Since ``Engine`` parameters are manually added, any change to the
            ``Engine`` class should be reflected manually.
        """
        from anomalib.callbacks.normalization import get_normalization_callback

        parser.add_function_arguments(get_normalization_callback, "normalization")
        parser.add_argument("--task", type=TaskType | str, default=TaskType.SEGMENTATION)
        parser.add_argument("--metrics.image", type=list[str] | str | None, default=["F1Score", "AUROC"])
        parser.add_argument("--metrics.pixel", type=list[str] | str | None, default=None, required=False)
        parser.add_argument("--metrics.threshold", type=Threshold | str, default="F1AdaptiveThreshold")
        parser.add_argument("--logging.log_graph", type=bool, help="Log the model to the logger", default=False)
        if hasattr(parser, "subcommand") and parser.subcommand not in {"export", "predict"}:
            parser.link_arguments("task", "data.init_args.task")
        parser.add_argument(
            "--default_root_dir",
            type=Path,
            help="Path to save the results.",
            default=Path("./results"),
        )
        parser.link_arguments("default_root_dir", "trainer.default_root_dir")
        # TODO(ashwinvaidya17): Tiling should also be a category of its own
        # CVS-122659

    def add_trainer_arguments(self, parser: ArgumentParser, subcommand: str) -> None:
        """Add train arguments to the parser."""
        self._add_default_arguments_to_parser(parser)
        self._add_trainer_arguments_to_parser(parser, add_optimizer=True, add_scheduler=True)
        parser.add_subclass_arguments(
            AnomalyModule,
            "model",
            fail_untyped=False,
            required=True,
        )
        parser.add_subclass_arguments(AnomalibDataModule, "data")
        self.add_arguments_to_parser(parser)
        skip: set[str | int] = set(self.subcommands()[subcommand])
        added = parser.add_method_arguments(
            Trainer,
            subcommand,
            skip=skip,
        )
        self.subcommand_method_arguments[subcommand] = added

    def add_train_arguments(self, parser: ArgumentParser) -> None:
        """Add train arguments to the parser."""
        self._add_default_arguments_to_parser(parser)
        self._add_trainer_arguments_to_parser(parser, add_optimizer=True, add_scheduler=True)
        parser.add_subclass_arguments(
            AnomalyModule,
            "model",
            fail_untyped=False,
            required=True,
        )
        parser.add_subclass_arguments(AnomalibDataModule, "data")
        self.add_arguments_to_parser(parser)
        added = parser.add_method_arguments(
            Engine,
            "train",
            skip={"model", "datamodule", "val_dataloaders", "test_dataloaders", "train_dataloaders"},
        )
        self.subcommand_method_arguments["train"] = added

    def add_predict_arguments(self, parser: ArgumentParser) -> None:
        """Add predict arguments to the parser."""
        self._add_default_arguments_to_parser(parser)
        self._add_trainer_arguments_to_parser(parser)
        parser.add_subclass_arguments(
            AnomalyModule,
            "model",
            fail_untyped=False,
            required=True,
        )
        parser.add_argument(
            "--data",
            type=Dataset | AnomalibDataModule | DataLoader | str | Path,
            required=True,
        )
        added = parser.add_method_arguments(
            Engine,
            "predict",
            skip={"model", "dataloaders", "datamodule", "dataset", "data_path"},
        )
        self.subcommand_method_arguments["predict"] = added
        self.add_arguments_to_parser(parser)

    def add_export_arguments(self, parser: ArgumentParser) -> None:
        """Add export arguments to the parser."""
        self._add_default_arguments_to_parser(parser)
        self._add_trainer_arguments_to_parser(parser)
        parser.add_subclass_arguments(
            AnomalyModule,
            "model",
            fail_untyped=False,
            required=True,
        )
        parser.add_argument(
            "--data",
            type=AnomalibDataModule,
            required=False,
        )
        added = parser.add_method_arguments(
            Engine,
            "export",
            skip={"ov_args", "model", "datamodule"},
        )
        self.subcommand_method_arguments["export"] = added
        add_openvino_export_arguments(parser)
        self.add_arguments_to_parser(parser)

    def _set_install_subcommand(self, action_subcommand: _ActionSubCommands) -> None:
        sub_parser = ArgumentParser(formatter_class=CustomHelpFormatter)
        sub_parser.add_argument(
            "--option",
            help="Install the full or optional-dependencies.",
            default="full",
            type=str,
            choices=["full", "core", "dev", "loggers", "notebooks", "openvino"],
        )
        sub_parser.add_argument(
            "-v",
            "--verbose",
            help="Set Logger level to INFO",
            action="store_true",
        )

        self.subcommand_parsers["install"] = sub_parser
        action_subcommand.add_subcommand(
            "install",
            sub_parser,
            help="Install the full-package for anomalib.",
        )

    def before_instantiate_classes(self) -> None:
        """Modify the configuration to properly instantiate classes and sets up tiler."""
        subcommand = self.config["subcommand"]
        if subcommand in {*self.subcommands(), "train", "predict"}:
            self.config[subcommand] = update_config(self.config[subcommand])

    def instantiate_classes(self) -> None:
        """Instantiate classes depending on the subcommand.

        For trainer related commands it instantiates all the model, datamodule and trainer classes.
        But for subcommands we do not want to instantiate any trainer specific classes such as datamodule, model, etc
        This is because the subcommand is responsible for instantiating and executing code based on the passed config
        """
        if self.config["subcommand"] in {*self.subcommands(), "predict"}:  # trainer commands
            # since all classes are instantiated, the LightningCLI also creates an unused ``Trainer`` object.
            # the minor change here is that engine is instantiated instead of trainer
            self.config_init = self.parser.instantiate_classes(self.config)
            self.datamodule = self._get(self.config_init, "data")
            if isinstance(self.datamodule, Dataset):
                self.datamodule = DataLoader(self.datamodule)
            self.model = self._get(self.config_init, "model")
            self._configure_optimizers_method_to_model()
            self.instantiate_engine()
        else:
            self.config_init = self.parser.instantiate_classes(self.config)
            subcommand = self.config["subcommand"]
            if subcommand in {"train", "export"}:
                self.instantiate_engine()
            if "model" in self.config_init[subcommand]:
                self.model = self._get(self.config_init, "model")
            else:
                self.model = None
            if "data" in self.config_init[subcommand]:
                self.datamodule = self._get(self.config_init, "data")
            else:
                self.datamodule = None

    def instantiate_engine(self) -> None:
        """Instantiate the engine.

        .. note::
            Most of the code in this method is taken from ``LightningCLI``'s
            ``instantiate_trainer`` method. Refer to that method for more
            details.
        """
        from lightning.pytorch.cli import SaveConfigCallback

        from anomalib.callbacks import get_callbacks

        engine_args = {
            "normalization": self._get(self.config_init, "normalization.normalization_method"),
            "threshold": self._get(self.config_init, "metrics.threshold"),
            "task": self._get(self.config_init, "task"),
            "image_metrics": self._get(self.config_init, "metrics.image"),
            "pixel_metrics": self._get(self.config_init, "metrics.pixel"),
        }
        trainer_config = {**self._get(self.config_init, "trainer", default={}), **engine_args}
        key = "callbacks"
        if key in trainer_config:
            if trainer_config[key] is None:
                trainer_config[key] = []
            elif not isinstance(trainer_config[key], list):
                trainer_config[key] = [trainer_config[key]]
            if not trainer_config.get("fast_dev_run", False):
                config_callback = SaveConfigCallback(
                    self._parser(self.subcommand),
                    self.config.get(str(self.subcommand), self.config),
                    overwrite=True,
                )
                trainer_config[key].append(config_callback)
        trainer_config[key].extend(get_callbacks(self.config[self.subcommand]))
        self.engine = Engine(**trainer_config)

    def _run_subcommand(self) -> None:
        """Run subcommand depending on the subcommand.

        This overrides the original ``_run_subcommand`` to run the ``Engine``
        method rather than the ``Train`` method.
        """
        if self.subcommand == "install":
            from anomalib.cli.install import anomalib_install

            install_kwargs = self.config.get("install", {})
            anomalib_install(**install_kwargs)
        elif self.config["subcommand"] in {*self.subcommands(), "train", "export", "predict"}:
            fn = getattr(self.engine, self.subcommand)
            fn_kwargs = self._prepare_subcommand_kwargs(self.subcommand)
            fn(**fn_kwargs)
        elif PIPELINE_REGISTRY is not None and self.subcommand in pipeline_subcommands():
            run_pipeline(self.config)
        else:
            self.config_init = self.parser.instantiate_classes(self.config)
            getattr(self, f"{self.subcommand}")()

    @property
    def fit(self) -> Callable:
        """Fit the model using engine's fit method."""
        return self.engine.fit

    @property
    def validate(self) -> Callable:
        """Validate the model using engine's validate method."""
        return self.engine.validate

    @property
    def test(self) -> Callable:
        """Test the model using engine's test method."""
        return self.engine.test

    @property
    def predict(self) -> Callable:
        """Predict using engine's predict method."""
        return self.engine.predict

    @property
    def train(self) -> Callable:
        """Train the model using engine's train method."""
        return self.engine.train

    @property
    def export(self) -> Callable:
        """Export the model using engine's export method."""
        return self.engine.export

    @staticmethod
    def _add_trainer_arguments_to_parser(
        parser: ArgumentParser,
        add_optimizer: bool = False,
        add_scheduler: bool = False,
    ) -> None:
        """Add trainer arguments to the parser."""
        parser.add_class_arguments(Trainer, "trainer", fail_untyped=False, instantiate=False, sub_configs=True)

        if add_optimizer:
            from torch.optim import Optimizer

            optim_kwargs = {"instantiate": False, "fail_untyped": False, "skip": {"params"}}
            parser.add_subclass_arguments(
                baseclass=(Optimizer,),
                nested_key="optimizer",
                **optim_kwargs,
            )
        if add_scheduler:
            from lightning.pytorch.cli import LRSchedulerTypeTuple

            scheduler_kwargs = {"instantiate": False, "fail_untyped": False, "skip": {"optimizer"}}
            parser.add_subclass_arguments(
                baseclass=LRSchedulerTypeTuple,
                nested_key="lr_scheduler",
                **scheduler_kwargs,
            )

    @staticmethod
    def _add_default_arguments_to_parser(parser: ArgumentParser) -> None:
        """Adds default arguments to the parser."""
        parser.add_argument(
            "--seed_everything",
            type=bool | int,
            default=True,
            help=(
                "Set to an int to run seed_everything with this value before classes instantiation."
                "Set to True to use a random seed."
            ),
        )

    def _get(self, config: Namespace, key: str, default: Any = None) -> Any:  # noqa: ANN401
        """Utility to get a config value which might be inside a subcommand."""
        return config.get(str(self.subcommand), config).get(key, default)

    def _prepare_subcommand_kwargs(self, subcommand: str) -> dict[str, Any]:
        """Prepares the keyword arguments to pass to the subcommand to run."""
        fn_kwargs = {
            k: v for k, v in self.config_init[subcommand].items() if k in self.subcommand_method_arguments[subcommand]
        }
        fn_kwargs["model"] = self.model
        if self.datamodule is not None:
            if isinstance(self.datamodule, AnomalibDataModule):
                fn_kwargs["datamodule"] = self.datamodule
            elif isinstance(self.datamodule, DataLoader):
                fn_kwargs["dataloaders"] = self.datamodule
            elif isinstance(self.datamodule, Path | str):
                fn_kwargs["data_path"] = self.datamodule
        return fn_kwargs

    def _parser(self, subcommand: str | None) -> ArgumentParser:
        if subcommand is None:
            return self.parser
        # return the subcommand parser for the subcommand passed
        return self.subcommand_parsers[subcommand]

    def _configure_optimizers_method_to_model(self) -> None:
        from lightning.pytorch.cli import LightningCLI, instantiate_class

        optimizer_cfg = self._get(self.config_init, "optimizer", None)
        if optimizer_cfg is None:
            return
        lr_scheduler_cfg = self._get(self.config_init, "lr_scheduler", {})

        optimizer = instantiate_class(self.model.parameters(), optimizer_cfg)
        lr_scheduler = instantiate_class(optimizer, lr_scheduler_cfg) if lr_scheduler_cfg else None
        fn = partial(LightningCLI.configure_optimizers, optimizer=optimizer, lr_scheduler=lr_scheduler)

        # override the existing method
        self.model.configure_optimizers = MethodType(fn, self.model)


def main() -> None:
    """Trainer via Anomalib CLI."""
    configure_logger()
    AnomalibCLI()


if __name__ == "__main__":
    main()
