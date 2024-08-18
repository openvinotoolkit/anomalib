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
from .components.utils import NormalizationStage, PredictData
from .components.utils.ensemble_engine import TiledEnsembleEngine

logger = logging.getLogger(__name__)


class TrainTiledEnsemble(Pipeline):
    """Tiled ensemble training pipeline."""

    def __init__(self) -> None:
        self.root_dir: Path

    def _setup_runners(self, args: dict) -> list[Runner]:
        """Setup the runners for the pipeline.

        This pipeline consists of training and validation steps:
        Training models > prediction on val data > merging val data >
        > (optionally) smoothing seams > calculation of post-processing statistics

        Returns:
            list[Runner]: List of runners executing tiled ensemble train + val jobs.
        """
        runners: list[Runner] = []
        self.root_dir = TiledEnsembleEngine.setup_ensemble_workspace(args)

        seed = args["seed"]
        accelerator = args["accelerator"]
        tiling_args = args["tiling"]
        data_args = args["data"]
        normalization_stage = NormalizationStage(args["normalization_stage"])
        model_args = args["TrainModels"]["model"]

        if accelerator == "cuda":
            runners.extend(
                [
                    ParallelRunner(
                        TrainModelJobGenerator(
                            seed=seed,
                            accelerator=accelerator,
                            root_dir=self.root_dir,
                            tiling_args=tiling_args,
                            data_args=data_args,
                            normalization_stage=normalization_stage,
                        ),
                        n_jobs=torch.cuda.device_count(),
                    ),
                    ParallelRunner(
                        PredictJobGenerator(
                            data_source=PredictData.VAL,
                            seed=seed,
                            accelerator=accelerator,
                            root_dir=self.root_dir,
                            tiling_args=tiling_args,
                            data_args=data_args,
                            model_args=model_args,
                            normalization_stage=normalization_stage,
                        ),
                        n_jobs=torch.cuda.device_count(),
                    ),
                ],
            )
        else:
            runners.extend(
                [
                    SerialRunner(
                        TrainModelJobGenerator(
                            seed=seed,
                            accelerator=accelerator,
                            root_dir=self.root_dir,
                            tiling_args=tiling_args,
                            data_args=data_args,
                            normalization_stage=normalization_stage,
                        ),
                    ),
                    SerialRunner(
                        PredictJobGenerator(
                            data_source=PredictData.VAL,
                            seed=seed,
                            accelerator=accelerator,
                            root_dir=self.root_dir,
                            tiling_args=tiling_args,
                            data_args=data_args,
                            model_args=model_args,
                            normalization_stage=normalization_stage,
                        ),
                    ),
                ],
            )
        runners.append(SerialRunner(MergeJobGenerator(tiling_args=tiling_args, data_args=data_args)))

        if args["SeamSmoothing"]["apply"]:
            runners.append(
                SerialRunner(
                    SmoothingJobGenerator(accelerator=accelerator, tiling_args=tiling_args, data_args=data_args),
                ),
            )

        runners.append(SerialRunner(StatisticsJobGenerator(self.root_dir)))

        return runners
