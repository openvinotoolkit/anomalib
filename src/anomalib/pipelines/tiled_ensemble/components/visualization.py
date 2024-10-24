"""Tiled ensemble - visualization job."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

from tqdm import tqdm

from anomalib import TaskType
from anomalib.data.utils.image import save_image
from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.tiled_ensemble.components.utils import NormalizationStage
from anomalib.pipelines.types import GATHERED_RESULTS, RUN_RESULTS
from anomalib.utils.visualization import ImageVisualizer

logger = logging.getLogger(__name__)


class VisualizationJob(Job):
    """Job for visualization of predictions.

    Args:
        predictions (list[Any]): list of image-level predictions.
        root_dir (Path): Root directory to save checkpoints, stats and images.
        task (TaskType): type of task the predictions represent.
        normalize (bool): if predictions need to be normalized
    """

    name = "Visualize"

    def __init__(self, predictions: list[Any], root_dir: Path, task: TaskType, normalize: bool) -> None:
        super().__init__()
        self.predictions = predictions
        self.root_dir = root_dir / "images"
        self.task = task
        self.normalize = normalize

    def run(self, task_id: int | None = None) -> list[Any]:
        """Run job that visualizes all prediction data.

        Args:
            task_id: Not used in this case.

        Returns:
            list[Any]: Unchanged predictions.
        """
        del task_id  # not needed here

        visualizer = ImageVisualizer(task=self.task, normalize=self.normalize)

        logger.info("Starting visualization.")

        for data in tqdm(self.predictions, desc="Visualizing"):
            for result in visualizer(outputs=data):
                # Finally image path is root/defect_type/image_name
                if result.file_name is not None:
                    file_path = Path(result.file_name)
                else:
                    msg = "file_path should exist in returned Visualizer."
                    raise ValueError(msg)

                root = self.root_dir / file_path.parent.name
                filename = file_path.name

                save_image(image=result.image, root=root, filename=filename)

        return self.predictions

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: Unchanged list of predictions.
        """
        # take the first element as result is list of lists here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """This job doesn't save anything."""


class VisualizationJobGenerator(JobGenerator):
    """Generate VisualizationJob.

    Args:
        root_dir (Path): Root directory where images will be saved (root/images).
    """

    def __init__(self, root_dir: Path, task: TaskType, normalization_stage: NormalizationStage) -> None:
        self.root_dir = root_dir
        self.task = task
        self.normalize = normalization_stage == NormalizationStage.NONE

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return VisualizationJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: list[Any] | None = None,
    ) -> Generator[VisualizationJob, None, None]:
        """Return a generator producing a single visualization job.

        Args:
            args: Ensemble run args.
            prev_stage_result (list[Any]): Ensemble predictions from previous step.

        Returns:
            Generator[VisualizationJob, None, None]: VisualizationJob generator
        """
        del args  # args not used here

        if prev_stage_result is not None:
            yield VisualizationJob(prev_stage_result, self.root_dir, self.task, self.normalize)
        else:
            msg = "Visualization job requires tile level predictions from previous step."
            raise ValueError(msg)
