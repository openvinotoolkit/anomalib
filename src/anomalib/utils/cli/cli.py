"""Anomalib CLI."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

from pytorch_lightning.cli import LightningArgumentParser, LightningCLI

from anomalib.models import AnomalyModule
from anomalib.trainer.trainer import AnomalibTrainer
from anomalib.utils.loggers import configure_logger

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
            **kwargs,
        )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add default arguments.

        Args:
            parser (LightningArgumentParser): Lightning Argument Parser.
        """
        add_post_processing_arguments(parser)
        add_metrics_arguments(parser)
        add_visualization_arguments(parser)
        add_logging_arguments(parser)

    def before_instantiate_classes(self) -> None:
        # TODO
        # loggers = get_experiment_logger()
        pass


def main() -> None:
    """Trainer via Anomalib CLI."""
    configure_logger()
    AnomalibCLI()


if __name__ == "__main__":
    main()
