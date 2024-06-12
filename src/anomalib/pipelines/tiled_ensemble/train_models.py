"""Tiled ensemble - ensemble training job."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Generator
from itertools import product
from pathlib import Path

from lightning import seed_everything

from anomalib.data import AnomalibDataModule
from anomalib.models import AnomalyModule
from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.types import GATHERED_RESULTS, PREV_STAGE_RESULT
from anomalib.utils.logging import hide_output
from anomalib.utils.normalization import NormalizationMethod

from .components.ensemble_engine import TiledEnsembleEngine
from .components.ensemble_tiling import EnsembleTiler
from .components.helper_functions import get_ensemble_datamodule, get_ensemble_model
from .components.post_processing.postprocess import NormalizationStage

logger = logging.getLogger(__name__)


class TrainModelJob(Job):
    """Job for training of individual models in the tiled ensemble.

    Args:
        accelerator (str): Accelerator (device) to use.
        seed (int): Random seed for reproducibility.
        root_dir (Path): Root directory to save checkpoints, stats and images.
        model (AnomalyModule): Model to train.
        datamodule (AnomalibDataModule): Datamodule with all dataloaders.
        tile_index (tuple[int, int]): Index of tile that this model processes.
        post_process_config (dict): Config dictionary for ensemble post-processing.

    """

    name = "pipeline"

    def __init__(
        self,
        accelerator: str,
        seed: int,
        root_dir: Path,
        model: AnomalyModule,
        datamodule: AnomalibDataModule,
        tile_index: tuple[int, int],
        post_process_config: dict,
    ) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.seed = seed
        self.root_dir = root_dir
        self.model = model
        self.datamodule = datamodule
        self.tile_index = tile_index
        self.post_process_config = post_process_config

    @hide_output
    def run(
        self,
        task_id: int | None = None,
    ) -> TiledEnsembleEngine:
        """Run train job that fits the model for given tile location.

        Args:
            task_id: Passed when job is ran in parallel

        Returns:
            TiledEnsembleEngine: engine with trained model.
        """
        devices: str | list[int] = "auto"
        if task_id is not None:
            devices = [task_id]
            logger.info(f"Running job {self.model.__class__.__name__} with device {task_id}")

        logger.info("Start of procedure for tile at position %s,", self.tile_index)
        seed_everything(self.seed)

        # if we want tile level normalization we set it here, otherwise it's done later on joined images
        if self.post_process_config["normalization_stage"] == NormalizationStage.TILE:
            normalization = NormalizationMethod.MIN_MAX
        else:
            normalization = NormalizationMethod.NONE

        # create engine for specific tile location and fit the model
        engine = TiledEnsembleEngine(
            tile_index=self.tile_index,
            normalization=normalization,
            accelerator=self.accelerator,
            devices=devices,
            default_root_dir=self.root_dir,
        )
        engine.fit(self.model, self.datamodule)
        # move model to cpu to avoid memory issues as the engine is returned to be used in validation phase
        self.model.cpu()

        return engine

    @staticmethod
    def collect(results: list[TiledEnsembleEngine]) -> dict[tuple[int, int], TiledEnsembleEngine]:
        """Collect engines from each tile location into a dict.

        Returns:
            dict[tuple[int, int], TiledEnsembleEngine]: Dict has form {tile_index: TiledEnsembleEngine}
        """
        return {r.tile_index: r for r in results}

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Skip as checkpoints are already saved by callback."""


class TrainModelJobGenerator(JobGenerator):
    """Generator for training job that train model for each tile location.

    Args:
        root_dir (Path): Root directory to save checkpoints, stats and images.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return TrainModelJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: PREV_STAGE_RESULT = None,
    ) -> Generator[Job, None, None]:
        """Generate training jobs for each tile location.

        Args:
            args (dict): Dict with config passed to training.
            prev_stage_result (None): not used here
        """
        del prev_stage_result  # Not needed for this job

        ensemble_args = args["ensemble"]
        model_args = args["model"]
        data_args = args["data"]

        # tiler used for splitting the image and getting the tile count
        tiler = EnsembleTiler(
            tile_size=ensemble_args["tiling"]["tile_size"],
            stride=ensemble_args["tiling"]["stride"],
            image_size=data_args["init_args"]["image_size"],
        )

        logger.info(
            "Tiled ensemble training started. Separate models will be trained for %d tile locations.",
            tiler.num_tiles,
        )
        # go over all tile positions
        for tile_index in product(range(tiler.num_patches_h), range(tiler.num_patches_w)):
            # prepare datamodule with custom collate function that only provides specific tile of image
            datamodule = get_ensemble_datamodule(data_args, tiler, tile_index)
            model = get_ensemble_model(model_args, tiler)

            # pass root_dir to engine so all models in ensemble have the same root dir
            yield TrainModelJob(
                accelerator=args["accelerator"],
                seed=args["seed"],
                root_dir=self.root_dir,
                model=model,
                datamodule=datamodule,
                tile_index=tile_index,
                post_process_config=ensemble_args["post_processing"],
            )
