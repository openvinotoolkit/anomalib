"""Tiled ensemble training pipeline."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from anomalib.pipelines.components.base import Pipeline, Runner
from anomalib.pipelines.components.runners import ParallelRunner, SerialRunner

from .components.ensemble_engine import TiledEnsembleEngine
from .predict import PredictData, PredictJobGenerator
from .train_models import TrainModelJobGenerator


class TrainTiledEnsemble(Pipeline):
    """Tiled ensemble training pipeline."""

    def _setup_runners(self, args: dict) -> list[Runner]:
        """Setup the runners for the pipeline."""
        runners: list[Runner] = []
        root_dir = TiledEnsembleEngine.setup_ensemble_workspace(args["pipeline"])

        if args["pipeline"]["accelerator"] == "cuda":
            runners.append(ParallelRunner(TrainModelJobGenerator(root_dir), n_jobs=torch.cuda.device_count()))
            runners.append(
                ParallelRunner(PredictJobGenerator(root_dir, PredictData.VAL), n_jobs=torch.cuda.device_count()),
            )
        else:
            runners.append(SerialRunner(TrainModelJobGenerator(root_dir)))
            runners.append(SerialRunner(PredictJobGenerator(root_dir, PredictData.VAL)))
        return runners
