"""Tiled ensemble test pipeline."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

import torch

from anomalib.data.utils import TestSplitMode
from anomalib.pipelines.components.base import Pipeline, Runner
from anomalib.pipelines.components.runners import ParallelRunner, SerialRunner
from anomalib.pipelines.tiled_ensemble.components import (
    MergeJobGenerator,
    MetricsCalculationJobGenerator,
    NormalizationJobGenerator,
    PredictJobGenerator,
    SmoothingJobGenerator,
    ThresholdingJobGenerator,
    VisualizationJobGenerator,
)
from anomalib.pipelines.tiled_ensemble.components.utils import NormalizationStage, PredictData, ThresholdStage

logger = logging.getLogger(__name__)


class EvalTiledEnsemble(Pipeline):
    """Tiled ensemble evaluation pipeline.

    Args:
        root_dir (Path): Path to root dir of run that contains checkpoints.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)

    def _setup_runners(self, args: dict) -> list[Runner]:
        """Set up the runners for the pipeline.

        This pipeline consists of jobs used to test/evaluate tiled ensemble:
        Prediction on test data > merging of predictions > (optional) seam smoothing
        > (optional) Normalization > (optional) Thresholding
        > Visualisation of predictions > Metrics calculation.

        Returns:
            list[Runner]: List of runners executing tiled ensemble testing jobs.
        """
        runners: list[Runner] = []

        if args["data"]["init_args"]["test_split_mode"] == TestSplitMode.NONE:
            logger.info("Test split mode set to `none`, skipping test phase.")
            return runners

        seed = args["seed"]
        accelerator = args["accelerator"]
        tiling_args = args["tiling"]
        data_args = args["data"]
        normalization_stage = NormalizationStage(args["normalization_stage"])
        threshold_stage = ThresholdStage(args["thresholding"]["stage"])
        model_args = args["TrainModels"]["model"]
        task = args["data"]["init_args"]["task"]
        metrics = args["TrainModels"]["metrics"]

        predict_job_generator = PredictJobGenerator(
            PredictData.TEST,
            seed=seed,
            accelerator=accelerator,
            root_dir=self.root_dir,
            tiling_args=tiling_args,
            data_args=data_args,
            model_args=model_args,
            normalization_stage=normalization_stage,
        )
        # 1. predict using test data
        if accelerator == "cuda":
            runners.append(
                ParallelRunner(
                    predict_job_generator,
                    n_jobs=torch.cuda.device_count(),
                ),
            )
        else:
            runners.append(
                SerialRunner(
                    predict_job_generator,
                ),
            )
        # 2. merge predictions
        runners.append(SerialRunner(MergeJobGenerator(tiling_args=tiling_args, data_args=data_args)))

        # 3. (optional) smooth seams
        if args["SeamSmoothing"]["apply"]:
            runners.append(
                SerialRunner(
                    SmoothingJobGenerator(accelerator=accelerator, tiling_args=tiling_args, data_args=data_args),
                ),
            )

        # 4. (optional) normalize
        if normalization_stage == NormalizationStage.IMAGE:
            runners.append(SerialRunner(NormalizationJobGenerator(self.root_dir)))
        # 5. (optional) threshold to get labels from scores
        if threshold_stage == ThresholdStage.IMAGE:
            runners.append(SerialRunner(ThresholdingJobGenerator(self.root_dir, normalization_stage)))

        # 6. visualize predictions
        runners.append(
            SerialRunner(VisualizationJobGenerator(self.root_dir, task=task, normalization_stage=normalization_stage)),
        )
        # calculate metrics
        runners.append(
            SerialRunner(
                MetricsCalculationJobGenerator(
                    accelerator=accelerator,
                    root_dir=self.root_dir,
                    task=task,
                    metrics=metrics,
                    normalization_stage=normalization_stage,
                ),
            ),
        )

        return runners
