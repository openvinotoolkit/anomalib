"""Tiled ensemble training pipeline."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from anomalib.data.utils import ValSplitMode

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
        thresholding_method = args["thresholding"]["method"]
        model_args = args["TrainModels"]["model"]

        train_job_generator = TrainModelJobGenerator(
            seed=seed,
            accelerator=accelerator,
            root_dir=self.root_dir,
            tiling_args=tiling_args,
            data_args=data_args,
            normalization_stage=normalization_stage,
        )

        predict_job_generator = PredictJobGenerator(
            data_source=PredictData.VAL,
            seed=seed,
            accelerator=accelerator,
            root_dir=self.root_dir,
            tiling_args=tiling_args,
            data_args=data_args,
            model_args=model_args,
            normalization_stage=normalization_stage,
        )

        # 1. train
        if accelerator == "cuda":
            runners.append(
                ParallelRunner(
                    train_job_generator,
                    n_jobs=torch.cuda.device_count(),
                ),
            )
        else:
            runners.append(
                SerialRunner(
                    train_job_generator,
                ),
            )

        if data_args["init_args"]["val_split_mode"] == ValSplitMode.NONE:
            logger.warning("No validation set provided, skipping statistics calculation.")
            return runners

        # 2. predict using validation data
        if accelerator == "cuda":
            runners.append(
                ParallelRunner(predict_job_generator, n_jobs=torch.cuda.device_count()),
            )
        else:
            runners.append(
                SerialRunner(predict_job_generator),
            )

        # 3. merge predictions
        runners.append(SerialRunner(MergeJobGenerator(tiling_args=tiling_args, data_args=data_args)))

        # 4. (optional) smooth seams
        if args["SeamSmoothing"]["apply"]:
            runners.append(
                SerialRunner(
                    SmoothingJobGenerator(accelerator=accelerator, tiling_args=tiling_args, data_args=data_args),
                ),
            )

        # 5. calculate statistics used for inference
        runners.append(SerialRunner(StatisticsJobGenerator(self.root_dir, thresholding_method)))

        return runners
