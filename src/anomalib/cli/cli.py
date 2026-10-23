# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomalib Command Line Interface.

This module provides the `AnomalibCLI` class for configuring and running Anomalib from the command line.
The CLI supports configuration via both command line arguments and configuration files (.yaml or .json).
"""

import ast
import importlib
import logging
import sys
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from types import MethodType
from typing import Any

from jsonargparse import ActionConfigFile, ArgumentParser, Namespace
from jsonargparse._actions import _ActionSubCommands
from rich import traceback

from anomalib import __version__
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
    from anomalib.models import AnomalibModule
    from anomalib.utils.config import update_config

except ImportError as error:
    _LIGHTNING_AVAILABLE = False
    logger.warning(f"Import failed: {error}. Please install the required dependencies.")


class AnomalibCLI:
    """Implementation of a fully configurable CLI tool for Anomalib.

    This class provides a flexible command-line interface that can be configured through
    both CLI arguments and configuration files. It supports various subcommands for
    training, testing, and exporting models.

    Args:
        args (Sequence[str] | None): Command line arguments. Defaults to None.
        run (bool): Whether to run the subcommand immediately. Defaults to True.

    Examples:
        Run from command line:

        >>> import sys
        >>> sys.argv = ["anomalib", "train", "--model", "Padim", "--data", "MVTecAD"]

        Run programmatically:

        >>> from anomalib.cli import AnomalibCLI
        >>> cli = AnomalibCLI(["train", "--model", "Padim", "--data", "MVTecAD"], run=False)

    Note:
        The CLI supports both YAML and JSON configuration files. Configuration can be
        provided via both files and command line arguments simultaneously.
    """

    def __init__(self, args: Sequence[str] | None = None, run: bool = True) -> None:
        self.parser = self.init_parser()
        self.subcommand_parsers: dict[str, ArgumentParser] = {}
        self.subcommand_method_arguments: dict[str, list[str]] = {}
        self._pre_processor_kwargs: dict[str, Any] = {}
        self.add_subcommands()
        args = self._extract_pre_processor_args(args)
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
            "fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule", "weights_only"},
            "validate": {"model", "dataloaders", "datamodule", "weights_only"},
            "test": {"model", "dataloaders", "datamodule", "weights_only"},
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
        parser.add_argument("--logging.log_graph", type=bool, help="Log the model to the logger", default=False)
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
            AnomalibModule,
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
            AnomalibModule,
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
            AnomalibModule,
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
            AnomalibModule,
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
            skip={"ov_kwargs", "model", "datamodule"},
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
        patch_info = self._patch_configure_pre_processor() if self._pre_processor_kwargs else None
        try:
            self._instantiate_classes()
        finally:
            if patch_info is not None:
                self._unpatch_configure_pre_processor(patch_info)

    def _instantiate_classes(self) -> None:
        """Internal instantiation logic, called within the pre-processor patch context."""
        if self.config["subcommand"] in {*self.subcommands(), "predict"}:  # trainer commands
            # since all classes are instantiated, the LightningCLI also creates an unused ``Trainer`` object.
            # the minor change here is that engine is instantiated instead of trainer
            self.config_init = self.parser.instantiate_classes(self.config)
            self.datamodule = self._get(self.config_init, "data")
            if isinstance(self.datamodule, Dataset):
                self.datamodule = DataLoader(self.datamodule, collate_fn=self.datamodule.collate_fn, pin_memory=True)
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

        engine_args: dict[str, Any] = {}
        trainer_config = {**self._get(self.config_init, "trainer", default={}), **engine_args}
        key = "callbacks"
        if key in trainer_config:
            if trainer_config[key] is None:
                trainer_config[key] = []
            elif not isinstance(trainer_config[key], list):
                trainer_config[key] = [trainer_config[key]]
            if not trainer_config.get("fast_dev_run"):
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

    def _extract_pre_processor_args(self, args: Sequence[str] | None) -> list[str]:
        """Extract ``--model.pre_processor.*`` arguments before jsonargparse parsing.

        jsonargparse rejects ``--model.pre_processor.image_size`` because
        ``pre_processor`` is typed as ``nn.Module | bool``.  This method strips
        those arguments from the raw CLI list and stores them in
        ``self._pre_processor_kwargs`` so they can be forwarded to
        ``configure_pre_processor`` later.

        Args:
            args: Raw CLI arguments, or ``None`` to read from ``sys.argv``.

        Returns:
            list[str]: CLI arguments with ``--model.pre_processor.*`` entries removed.
        """
        if args is None:
            args = sys.argv[1:]
        args = list(args)

        prefix = "--model.pre_processor."
        indices_to_remove: list[int] = []
        for i, arg in enumerate(args):
            if not arg.startswith(prefix):
                continue
            if "=" in arg:
                # --model.pre_processor.image_size=512
                key, val = arg[len(prefix) :].split("=", 1)
                self._pre_processor_kwargs[key] = self._parse_cli_value(val)
                indices_to_remove.append(i)
            elif i + 1 < len(args):
                # --model.pre_processor.image_size 512
                key = arg[len(prefix) :]
                self._pre_processor_kwargs[key] = self._parse_cli_value(args[i + 1])
                indices_to_remove.extend([i, i + 1])

        # Convert int image_size to (h, w) tuple for consistency
        if "image_size" in self._pre_processor_kwargs:
            val = self._pre_processor_kwargs["image_size"]
            if isinstance(val, int):
                self._pre_processor_kwargs["image_size"] = (val, val)
            elif isinstance(val, list):
                self._pre_processor_kwargs["image_size"] = tuple(val)

        for idx in reversed(indices_to_remove):
            args.pop(idx)
        return args

    @staticmethod
    def _parse_cli_value(val: str) -> Any:  # noqa: ANN401
        """Parse a CLI string value into a Python object using literal evaluation."""
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return val

    def _patch_configure_pre_processor(self) -> tuple[type, bool, Any] | None:
        """Patch the model class's ``configure_pre_processor`` with extracted CLI kwargs.

        Uses ``functools.partial`` to bind the extracted pre-processor arguments
        (e.g. ``image_size``) to the model's ``configure_pre_processor`` method
        before the model is instantiated.  This follows the same pattern as
        Lightning CLI's ``configure_optimizers`` override.

        Returns:
            Tuple of (model_class, had_own_method, original_descriptor) for
            restoration, or ``None`` if patching was not applicable.
        """
        subcommand = self.config["subcommand"]
        model_cfg = self.config.get(subcommand, self.config).get("model")
        if model_cfg is None:
            return None

        class_path = model_cfg.get("class_path") if hasattr(model_cfg, "get") else getattr(model_cfg, "class_path", None)
        if class_path is None:
            return None

        # Resolve the model class
        module_path, class_name = class_path.rsplit(".", 1)
        model_class = getattr(importlib.import_module(module_path), class_name)

        # Save the original descriptor so we can restore it after instantiation
        has_own = "configure_pre_processor" in model_class.__dict__
        original_descriptor = model_class.__dict__.get("configure_pre_processor")

        # Get the resolved callable (handles both @staticmethod and @classmethod)
        callable_method = model_class.configure_pre_processor
        patched = partial(callable_method, **self._pre_processor_kwargs)
        model_class.configure_pre_processor = staticmethod(patched)

        return (model_class, has_own, original_descriptor)

    @staticmethod
    def _unpatch_configure_pre_processor(patch_info: tuple[type, bool, Any]) -> None:
        """Restore the original ``configure_pre_processor`` after model instantiation."""
        model_class, has_own, original_descriptor = patch_info
        if has_own:
            model_class.configure_pre_processor = original_descriptor
        else:
            delattr(model_class, "configure_pre_processor")

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
