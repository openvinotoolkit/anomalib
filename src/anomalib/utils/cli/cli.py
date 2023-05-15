"""Anomalib CLI."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
from pathlib import Path
from typing import Any

from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

from anomalib.models import AnomalyModule
from anomalib.trainer.trainer import AnomalibTrainer
from anomalib.utils.loggers import configure_logger

from .subcommands import add_benchmarking_parser, add_export_parser, add_hpo_parser
from .utils import (
    add_logging_arguments,
    add_metrics_arguments,
    add_post_processing_arguments,
    add_visualization_arguments,
)

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
        **kwargs: Any,  # Remove with deprecations of v2.0.0
    ) -> None:
        kwargs.pop("trainer_class", None)
        kwargs.pop("model_class", None)
        super().__init__(
            model_class=AnomalyModule,
            trainer_class=AnomalibTrainer,
            subclass_mode_model=kwargs.pop("subclass_mode_model", True),
            save_config_kwargs=kwargs.pop("save_config_kwargs", {"overwrite": True}),
            **kwargs,
        )

    def _add_subcommands(self, parser: LightningArgumentParser, **kwargs: Any) -> None:
        """Setup base subcommands and add anomalib specific on top of it."""
        # Initializes fit, validate, test, predict and tune
        super()._add_subcommands(parser, **kwargs)
        add_export_parser(parser)
        add_hpo_parser(parser)
        add_benchmarking_parser(parser)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add default arguments.

        Args:
            parser (LightningArgumentParser): Lightning Argument Parser.
        """
        add_post_processing_arguments(parser)
        add_metrics_arguments(parser)
        add_visualization_arguments(parser)
        add_logging_arguments(parser)
        # link other arguments
        parser.link_arguments("data.init_args.image_size", "model.init_args.input_size")

    def before_instantiate_classes(self) -> None:
        # TODO
        # loggers = get_experiment_logger()
        config = self.config[self.config.subcommand]
        self._set_default_root_dir(config)
        self._set_ckpt_path(config)

    def _set_default_root_dir(self, config):
        if config.trainer.default_root_dir is None:
            project_path = (
                Path("./results")
                / config.model.class_path.split(".")[-1].lower()
                / config.data.class_path.split(".")[-1].lower()
            )
            if config.data.init_args.get("category", None) is not None:
                project_path /= config.data.init_args["category"]
            # TODO make unique directory
            project_path /= "run"
            config.trainer.default_root_dir = str(project_path)

    def _set_ckpt_path(self, config):
        if config.ckpt_path is None and config.trainer.default_root_dir is not None:
            model_ckpt_path = Path(config.trainer.default_root_dir) / "weights" / "lightning" / "model.ckpt"
            if model_ckpt_path.exists():
                logger.info(f"Model checkpoint exists at {model_ckpt_path}. Setting ckpt_path to this.")
                config.ckpt_path = str(model_ckpt_path)


def main() -> None:
    """Trainer via Anomalib CLI."""
    configure_logger()
    AnomalibCLI()


if __name__ == "__main__":
    main()
