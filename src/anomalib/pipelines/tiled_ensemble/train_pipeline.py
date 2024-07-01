"""Tiled ensemble training pipeline."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import logging

import torch

from anomalib.pipelines.components.base import Pipeline, Runner
from anomalib.pipelines.components.runners import ParallelRunner, SerialRunner

from .components import (
    MergeJobGenerator,
    PredictJobGenerator,
    SmoothingJobGenerator,
    StatisticsJobGenerator,
    TrainModelJobGenerator,
)
from .components.utils import PredictData
from .components.utils.ensemble_engine import TiledEnsembleEngine

logger = logging.getLogger(__name__)


class TrainTiledEnsemble(Pipeline):
    """Tiled ensemble training pipeline."""

    def __init__(self) -> None:
        self.root_dir: Path | None = None

    def _setup_runners(self, args: dict) -> list[Runner]:
        """Setup the runners for the pipeline.

        This pipeline consists of training and validation steps:
        Training models > prediction on val data > merging val data >
        > (optionally) smoothing seams > calculation of post-processing statistics

        Returns:
            list[Runner]: List of runners executing tiled ensemble train + val jobs.
        """
        runners: list[Runner] = []
        self.root_dir = TiledEnsembleEngine.setup_ensemble_workspace(args["pipeline"])

        if args["pipeline"]["accelerator"] == "cuda":
            runners.extend(
                [
                    ParallelRunner(TrainModelJobGenerator(self.root_dir), n_jobs=torch.cuda.device_count()),
                    ParallelRunner(
                        PredictJobGenerator(self.root_dir, PredictData.VAL),
                        n_jobs=torch.cuda.device_count(),
                    ),
                ],
            )
        else:
            runners.extend(
                [
                    SerialRunner(TrainModelJobGenerator(self.root_dir)),
                    SerialRunner(PredictJobGenerator(self.root_dir, PredictData.VAL)),
                ],
            )
        runners.append(SerialRunner(MergeJobGenerator()))

        if args["pipeline"]["ensemble"]["post_processing"]["seam_smoothing"]["apply"]:
            runners.append(SerialRunner(SmoothingJobGenerator()))

        runners.append(SerialRunner(StatisticsJobGenerator(self.root_dir)))

        return runners
