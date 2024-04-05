"""Main pipeline runner."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Any

from rich import traceback

traceback.install()
log_file = "runs/pipeline.log"
Path(log_file).parent.mkdir(exist_ok=True, parents=True)
logger_file_handler = logging.FileHandler(log_file)
logging.getLogger().addHandler(logger_file_handler)
logging.getLogger().setLevel(logging.DEBUG)


from jsonargparse import ActionConfigFile, ArgumentParser, Namespace  # noqa: E402
from rich import print  # noqa: E402

from anomalib.pipelines.executors.serial import SerialExecutor  # noqa: E402

logger = logging.getLogger(__name__)
# TODO(ashwinvaidya17): capture all terminal output and pipe it to a file


class Pipeline:
    """Pipeline class."""

    def __init__(self, executors: list[SerialExecutor] | SerialExecutor) -> None:
        self.executors = executors if isinstance(executors, list) else [executors]

    def run(self, args: Namespace | None = None) -> Any:  # noqa: ANN401
        """Run the pipeline.

        Args:
            args (Namespace): Arguments to run the pipeline. These are the args returned by ArgumentParser.
        """
        if args is None:
            logger.warning("No arguments provided, parsing arguments from command line.")
            parser = self.get_parser()
            args = parser.parse_args()

        for executor in self.executors:
            try:
                executor.run(args)
            except Exception:  # noqa: PERF203 catch all exception and allow try-catch in loop
                logger.exception("An error occurred when running the executor.")
                print(
                    f"There were some errors when running [red]{executor.job.name}[/red] with"
                    f" [green]{executor.__class__.__name__}[/green]."
                    f" Please check [magenta]{log_file}[/magenta] for more details.",
                )

    def get_parser(self, parser: ArgumentParser | None = None) -> ArgumentParser:
        """Create a new parser if none is provided."""
        if parser is None:
            parser = ArgumentParser()
            parser.add_argument("--config", action=ActionConfigFile, help="Configuration file path.")

        # add executor specific arguments
        for executor in self.executors:
            executor.add_arguments(parser)
        return parser
