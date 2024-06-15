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


class TestTiledEnsemble(Pipeline):
    """Tiled ensemble testing pipeline.

    Args:
        root_dir (Path): Path to root dir of run that contains checkpoints.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def _setup_runners(self, args: dict) -> list[Runner]:
        """Set up the runners for the pipeline.

        This pipeline consists of jobs used to test/evaluate tiled ensemble:
        Prediction on test data > merging of predictions > (optional) seam smoothing
        > Normalization > Thresholding (if option for image-stage is used)
        > Visualisation of predictions > Metrics calculation.

        Returns:
            list[Runner]: List of runners executing tiled ensemble testing jobs.
        """
        runners: list[Runner] = []

        if args["pipeline"]["data"]["init_args"]["test_split_mode"] == TestSplitMode.NONE:
            logger.info("Test split mode set to `none`, skipping test phase.")
            return runners

        if args["pipeline"]["accelerator"] == "cuda":
            runners.append(
                ParallelRunner(PredictJobGenerator(self.root_dir, PredictData.TEST), n_jobs=torch.cuda.device_count()),
            )
        else:
            runners.append(
                SerialRunner(PredictJobGenerator(self.root_dir, PredictData.TEST)),
            )
        runners.append(SerialRunner(MergeJobGenerator()))

        if args["pipeline"]["ensemble"]["post_processing"]["seam_smoothing"]["apply"]:
            runners.append(SerialRunner(SmoothingJobGenerator()))

        if args["pipeline"]["ensemble"]["post_processing"]["normalization_stage"] == NormalizationStage.IMAGE:
            runners.append(SerialRunner(NormalizationJobGenerator(self.root_dir)))
        if args["pipeline"]["ensemble"]["post_processing"]["threshold_stage"] == ThresholdStage.IMAGE:
            runners.append(SerialRunner(ThresholdingJobGenerator(self.root_dir)))

        runners.append(SerialRunner(VisualizationJobGenerator(self.root_dir)))
        runners.append(SerialRunner(MetricsCalculationJobGenerator(self.root_dir)))

        return runners
