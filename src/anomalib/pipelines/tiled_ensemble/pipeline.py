"""Tiled ensemble training pipeline."""
import logging

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from anomalib.pipelines.components.base import Pipeline, Runner
from anomalib.pipelines.components.runners import ParallelRunner, SerialRunner

from .calculate_stats import StatisticsJobGenerator
from .components.ensemble_engine import TiledEnsembleEngine
from .merge import MergeJobGenerator
from .metric_calculation import MetricsCalculationJobGenerator
from .normalization import NormalizationJobGenerator, NormalizationStage
from .predict import PredictData, PredictJobGenerator
from .smoothing import SmoothingJobGenerator
from .threshold import ThresholdingJobGenerator, ThresholdStage
from .train_models import TrainModelJobGenerator
from .visualize import VisualizationJobGenerator
from ...data.utils import TestSplitMode


logger = logging.getLogger(__name__)


class TrainTiledEnsemble(Pipeline):
    """Tiled ensemble training pipeline."""

    def _setup_runners(self, args: dict) -> list[Runner]:
        """Setup the runners for the pipeline."""
        runners: list[Runner] = []
        root_dir = TiledEnsembleEngine.setup_ensemble_workspace(args["pipeline"])

        """==== TRAIN + VAL STEPS ===="""

        if args["pipeline"]["accelerator"] == "cuda":
            runners.extend(
                [
                    ParallelRunner(TrainModelJobGenerator(root_dir), n_jobs=torch.cuda.device_count()),
                    ParallelRunner(PredictJobGenerator(root_dir, PredictData.VAL), n_jobs=torch.cuda.device_count()),
                ],
            )
        else:
            runners.extend(
                [
                    SerialRunner(TrainModelJobGenerator(root_dir)),
                    SerialRunner(PredictJobGenerator(root_dir, PredictData.VAL)),
                ],
            )
        runners.append(SerialRunner(MergeJobGenerator()))

        if args["pipeline"]["ensemble"]["post_processing"]["seam_smoothing"]["apply"]:
            runners.append(SerialRunner(SmoothingJobGenerator()))

        runners.append(SerialRunner(StatisticsJobGenerator(root_dir)))

        """==== TEST STEPS ===="""

        if args["pipeline"]["data"]["init_args"]["test_split_mode"] == TestSplitMode.NONE:
            logger.info("Test split mode set to `none`, skipping test phase.")
            return runners

        if args["pipeline"]["accelerator"] == "cuda":
            runners.append(
                ParallelRunner(PredictJobGenerator(root_dir, PredictData.TEST), n_jobs=torch.cuda.device_count()),
            )
        else:
            runners.append(
                SerialRunner(PredictJobGenerator(root_dir, PredictData.TEST)),
            )
        runners.append(SerialRunner(MergeJobGenerator()))

        if args["pipeline"]["ensemble"]["post_processing"]["seam_smoothing"]["apply"]:
            runners.append(SerialRunner(SmoothingJobGenerator()))

        if args["pipeline"]["ensemble"]["post_processing"]["normalization_stage"] == NormalizationStage.IMAGE:
            runners.append(SerialRunner(NormalizationJobGenerator(root_dir)))
        if args["pipeline"]["ensemble"]["post_processing"]["threshold_stage"] == ThresholdStage.IMAGE:
            runners.append(SerialRunner(ThresholdingJobGenerator(root_dir)))

        runners.append(SerialRunner(VisualizationJobGenerator(root_dir)))
        runners.append(SerialRunner(MetricsCalculationJobGenerator(root_dir)))

        return runners
