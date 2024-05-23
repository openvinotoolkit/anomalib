"""HPO job."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from tempfile import TemporaryDirectory

from lightning.pytorch.loggers import Logger

from anomalib.data import AnomalibDataModule
from anomalib.engine import Engine
from anomalib.models import AnomalyModule
from anomalib.pipelines.components import Job
from anomalib.utils.logging import hide_output

logger = logging.getLogger(__name__)


class HPOJob(Job):
    """HPO job."""

    name = "hpo"

    def __init__(self, model: AnomalyModule, datamodule: AnomalibDataModule, logger: Logger) -> None:
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.logger = logger

    @hide_output
    def run(
        self,
        task_id: int | None = None,
    ) -> None:
        """Run the HPO."""
        del task_id  # Not needed for this job
        with TemporaryDirectory() as temp_dir:
            engine = Engine(
                default_root_dir=temp_dir,
                logger=self.logger,
            )
            engine.fit(self.model, datamodule=self.datamodule)
            self.logger.finalize(status="completed")

    @staticmethod
    def collect(_: list) -> None:
        """Does not collect any results."""
        return

    @staticmethod
    def save(_: None = None) -> None:
        """Does not save any results."""
        return
