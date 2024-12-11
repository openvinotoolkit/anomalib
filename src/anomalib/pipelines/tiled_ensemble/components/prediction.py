"""Tiled ensemble - ensemble prediction job."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Generator
from itertools import product
from pathlib import Path
from typing import Any

from lightning import seed_everything
from torch.utils.data import DataLoader

from anomalib.models import AnomalyModule
from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.types import GATHERED_RESULTS, PREV_STAGE_RESULT

from .utils import NormalizationStage, PredictData
from .utils.ensemble_engine import TiledEnsembleEngine
from .utils.helper_functions import (
    get_ensemble_datamodule,
    get_ensemble_engine,
    get_ensemble_model,
    get_ensemble_tiler,
)
from .utils.prediction_data import EnsemblePredictions

logger = logging.getLogger(__name__)


class PredictJob(Job):
    """Job for generating predictions with individual models in the tiled ensemble.

    Args:
        accelerator (str): Accelerator (device) to use.
        seed (int): Random seed for reproducibility.
        root_dir (Path): Root directory to save checkpoints, stats and images.
        tile_index (tuple[int, int]): Index of tile that this model processes.
        normalization_stage (str): Normalization stage flag.
        dataloader (DataLoader): Dataloader to use for training (either val or test).
        model (AnomalyModule): Model to train.
        engine (TiledEnsembleEngine | None):
            engine from train job. If job is used standalone, instantiate engine and model from checkpoint.
        ckpt_path (Path | None): Path to checkpoint to be loaded if engine doesn't contain correct weights.

    """

    name = "Predict"

    def __init__(
        self,
        accelerator: str,
        seed: int,
        root_dir: Path,
        tile_index: tuple[int, int],
        normalization_stage: str,
        dataloader: DataLoader,
        model: AnomalyModule | None,
        engine: TiledEnsembleEngine | None,
        ckpt_path: Path | None,
    ) -> None:
        super().__init__()
        if engine is None and ckpt_path is None:
            msg = "Either engine or checkpoint must be provided to predict job."
            raise ValueError(msg)

        self.accelerator = accelerator
        self.seed = seed
        self.root_dir = root_dir
        self.tile_index = tile_index
        self.normalization_stage = normalization_stage
        self.dataloader = dataloader
        self.model = model
        self.engine = engine
        self.ckpt_path = ckpt_path

    def run(
        self,
        task_id: int | None = None,
    ) -> tuple[tuple[int, int], Any | None]:
        """Predict job that predicts the data with specific model for given tile location.

        Args:
            task_id: Passed when job is ran in parallel.

        Returns:
            tuple[tuple[int, int], list[Any]]: Tile index, List of predictions.
        """
        devices: str | list[int] = "auto"
        if task_id is not None:
            devices = [task_id]
            logger.info(f"Running job {self.model.__class__.__name__} with device {task_id}")

        logger.info("Start of predicting for tile at position %s,", self.tile_index)
        seed_everything(self.seed)

        if self.engine is None:
            # in case predict is invoked separately from train job, make new engine instance
            self.engine = get_ensemble_engine(
                tile_index=self.tile_index,
                accelerator=self.accelerator,
                devices=devices,
                root_dir=self.root_dir,
                normalization_stage=self.normalization_stage,
            )

        predictions = self.engine.predict(model=self.model, dataloaders=self.dataloader, ckpt_path=self.ckpt_path)

        # also return tile index as it's needed in collect method
        return self.tile_index, predictions

    @staticmethod
    def collect(results: list[tuple[tuple[int, int], list[Any]]]) -> EnsemblePredictions:
        """Collect predictions from each tile location into the predictions class.

        Returns:
            EnsemblePredictions: Object containing all predictions in form ready for merging.
        """
        storage = EnsemblePredictions()

        for tile_index, predictions in results:
            storage.add_tile_prediction(tile_index, predictions)

        return storage

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """This stage doesn't save anything."""


class PredictJobGenerator(JobGenerator):
    """Generator for predict job that uses individual models to predict for each tile location.

    Args:
        root_dir (Path): Root directory to save checkpoints, stats and images.
        data_source (PredictData): Whether to predict on validation set. If false use test set.
    """

    def __init__(
        self,
        data_source: PredictData,
        seed: int,
        accelerator: str,
        root_dir: Path,
        tiling_args: dict,
        data_args: dict,
        model_args: dict,
        normalization_stage: NormalizationStage,
    ) -> None:
        self.data_source = data_source
        self.seed = seed
        self.accelerator = accelerator
        self.root_dir = root_dir
        self.tiling_args = tiling_args
        self.data_args = data_args
        self.model_args = model_args
        self.normalization_stage = normalization_stage

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return PredictJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: PREV_STAGE_RESULT = None,
    ) -> Generator[PredictJob, None, None]:
        """Generate predict jobs for each tile location.

        Args:
            args (dict): Dict with config passed to training.
            prev_stage_result (dict[tuple[int, int], TiledEnsembleEngine] | None):
                if called after train job this contains engines with individual models, otherwise load from checkpoints.

        Returns:
            Generator[PredictJob, None, None]: PredictJob generator.
        """
        del args  # args not used here

        # tiler used for splitting the image and getting the tile count
        tiler = get_ensemble_tiler(self.tiling_args, self.data_args)

        logger.info(
            "Tiled ensemble predicting started using %s data.",
            self.data_source.value,
        )
        # go over all tile positions
        for tile_index in product(range(tiler.num_patches_h), range(tiler.num_patches_w)):
            # prepare datamodule with custom collate function that only provides specific tile of image
            datamodule = get_ensemble_datamodule(self.data_args, tiler, tile_index)

            # check if predict step is positioned after training
            if prev_stage_result and tile_index in prev_stage_result:
                engine = prev_stage_result[tile_index]
                # model is inside engine in this case
                model = engine.model
                ckpt_path = None
            else:
                # any other case - predict is called standalone
                engine = None
                # we need to make new model instance as it's not inside engine
                model = get_ensemble_model(self.model_args, tiler)
                tile_i, tile_j = tile_index
                # prepare checkpoint path for model on current tile location
                ckpt_path = self.root_dir / "weights" / "lightning" / f"model{tile_i}_{tile_j}.ckpt"

            # pick the dataloader based on predict data
            dataloader = datamodule.test_dataloader()
            if self.data_source == PredictData.VAL:
                dataloader = datamodule.val_dataloader()
            # TODO(blaz-r): - this is tweak to avoid problem in engine:388
            # 2254
            dataloader.dataset.transform = None

            # pass root_dir to engine so all models in ensemble have the same root dir
            yield PredictJob(
                accelerator=self.accelerator,
                seed=self.seed,
                root_dir=self.root_dir,
                tile_index=tile_index,
                normalization_stage=self.normalization_stage,
                model=model,
                dataloader=dataloader,
                engine=engine,
                ckpt_path=ckpt_path,
            )
