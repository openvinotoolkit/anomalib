"""Tiled ensemble - seam smoothing job."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Generator
from typing import Any

import torch

from anomalib.models.components import GaussianBlur2d
from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.tiled_ensemble.components.ensemble_tiling import EnsembleTiler
from anomalib.pipelines.tiled_ensemble.components.helper_functions import get_ensemble_tiler
from anomalib.pipelines.types import GATHERED_RESULTS, RUN_RESULTS

logger = logging.getLogger(__name__)


class SmoothingJob(Job):
    """Job for smoothing the area around the tile seam.

    Args:
        predictions (list[Any]): list of image-level predictions.
        width_factor (float):  Factor multiplied by tile dimension to get the region around seam which will be smoothed.
        filter_sigma (float): Sigma of filter used for smoothing the seams.
        tiler (EnsembleTiler): Tiler object used to get tile dimension data.
    """

    name = "pipeline"

    def __init__(self, predictions: list[Any], width_factor: float, filter_sigma: float, tiler: EnsembleTiler) -> None:
        super().__init__()
        self.predictions = predictions

        # offset in pixels of region around tile seam that will be smoothed
        self.height_offset = int(tiler.tile_size_h * width_factor)
        self.width_offset = int(tiler.tile_size_w * width_factor)
        self.tiler = tiler

        self.seam_mask = self.prepare_seam_mask()

        self.blur = GaussianBlur2d(sigma=filter_sigma)

    def prepare_seam_mask(self) -> torch.Tensor:
        """Prepare boolean mask of regions around the part where tiles seam in ensemble.

        Returns:
            Tensor: Representation of boolean mask where filtered seams should be used.
        """
        img_h, img_w = self.tiler.image_size
        stride_h, stride_w = self.tiler.stride_h, self.tiler.stride_w

        mask = torch.zeros(img_h, img_w, dtype=torch.bool)

        # prepare mask strip on vertical seams
        curr_w = stride_w
        while curr_w < img_w:
            start_i = curr_w - self.width_offset
            end_i = curr_w + self.width_offset
            mask[:, start_i:end_i] = 1
            curr_w += stride_w

        # prepare mask strip on horizontal seams
        curr_h = stride_h
        while curr_h < img_h:
            start_i = curr_h - self.height_offset
            end_i = curr_h + self.height_offset
            mask[start_i:end_i, :] = True
            curr_h += stride_h

        return mask

    def run(self, task_id: int | None = None) -> list[Any]:
        """Run smoothing job.

        Args:
            task_id: not used in this case

        Returns:
            list[Any]: list of predictions.
        """
        del task_id  # not needed here

        for data in self.predictions:
            # smooth the anomaly map and take only region around seams
            smoothed = self.blur(data["anomaly_maps"])
            data["anomaly_maps"][:, :, self.seam_mask] = smoothed[:, :, self.seam_mask]

        return self.predictions

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: list of predictions.
        """
        # take the first element as result is list of lists here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Nothing to save in this job."""


class SmoothingJobGenerator(JobGenerator):
    """Generate SmoothingJob."""

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return SmoothingJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: list[Any] | None = None,
    ) -> Generator[Job, None, None]:
        """Return a generator producing a single seam smoothing job.

        Args:
            args: tiled ensemble pipeline args.
            prev_stage_result (list[Any]): ensemble predictions from merging step.

        Returns:
            Generator[Job, None, None]: MergeJob generator
        """
        tiler = get_ensemble_tiler(args)
        yield SmoothingJob(
            predictions=prev_stage_result,
            width_factor=args["ensemble"]["post_processing"]["seam_smoothing"]["width"],
            filter_sigma=args["ensemble"]["post_processing"]["seam_smoothing"]["width"],
            tiler=tiler,
        )
