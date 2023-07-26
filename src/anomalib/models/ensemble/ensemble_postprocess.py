"""Classes and functions used for various postprocessing of ensemble predictions."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC
from typing import Any, List

import torch
from tqdm import tqdm

from anomalib.models.ensemble.ensemble_prediction_data import EnsemblePredictions
from anomalib.models.ensemble.ensemble_prediction_joiner import EnsemblePredictionJoiner
from anomalib.utils.metrics import AnomalyScoreThreshold, MinMax

logger = logging.getLogger(__name__)


class EnsemblePostProcess(ABC):
    """
    Abstract class for use as building block of post-processing pipeline.

    Args:
        final_compute: Flag if this block produces values that are computed at the end.
        name: Name of block, that will be used in pipeline output dictionary.
    """

    def __init__(self, final_compute: bool, name: str) -> None:
        self.final_compute = final_compute
        self.name = name

    def process(self, data: Any) -> Any:
        """
        Process data as part of pipeline.

        Args:
            data: Prediction data to be processed.

        Returns:
            Processed data.
        """
        return data

    def compute(self) -> Any:
        """
        Once all data is processed, compute additional information.

        Returns:
            Computed data.
        """
        raise NotImplementedError


class SmoothJoins(EnsemblePostProcess):
    """
    Smooth the regions where tiles join in ensemble.

    """

    def __init__(self):
        super().__init__(final_compute=False, name="smooth_joins")


class MinMaxNormalize(EnsemblePostProcess):
    """
    Normalize images using min max normalization.

    Args:
        stats: dictionary containing statistics used for normalization (min, max, image threshold, pixel threshold).
    """

    def __init__(self, stats: dict[str, float]):
        super().__init__(final_compute=False, name="minmax_normalize")


class Threshold(EnsemblePostProcess):
    """
    Threshold the predictions using provided thresholds.

    Args:
        image_threshold: Threshold used for image-level thresholding.
        pixel_threshold: Threshold used for pixel-level thresholding.
    """

    def __init__(self, image_threshold: float, pixel_threshold: float):
        super().__init__(final_compute=False, name="threshold")


class PostProcessStats(EnsemblePostProcess):
    """
    Class used to obtain threshold and normalization statistics: (min, max, image threshold, pixel threshold).

    """

    def __init__(self):
        super().__init__(final_compute=True, name="stats")

        # adaptive threshold used for image and pixel level thresholding.
        self.image_threshold = AnomalyScoreThreshold().cpu()
        self.pixel_threshold = AnomalyScoreThreshold().cpu()
        self.pixel_update_called = False

        # used for minmax normalization
        self.minmax = MinMax().cpu()

    def process(self, data: Any) -> None:
        """
        Add current data to statistics accumulator.

        Args:
            data: Joined tile prediction data.
        """
        # update minmax
        if "anomaly_maps" in data:
            self.minmax(data["anomaly_maps"])
        elif "box_scores" in data:
            self.minmax(torch.cat(data["box_scores"]))
        elif "pred_scores" in data:
            self.minmax(data["pred_scores"])
        else:
            raise ValueError("No values found for normalization, provide anomaly maps, bbox scores, or image scores")

        # update thresholds
        self.image_threshold.update(data["pred_scores"], data["label"].int())
        if "mask" in data.keys() and "anomaly_maps" in data.keys():
            self.pixel_threshold.update(torch.squeeze(data["anomaly_maps"]), torch.squeeze(data["mask"].int()))
            self.pixel_update_called = True

    def compute(self) -> dict[str, float]:
        """
        At the end, compute actual values from all input data.

        Returns:
            Dictionary containing computed statistics: (min, max, image threshold, pixel threshold).
        """
        self.image_threshold.compute()
        if self.pixel_update_called:
            self.pixel_threshold.compute()
        else:
            self.pixel_threshold.value = self.image_threshold.value

        out = {
            "min": self.minmax.min.item(),
            "max": self.minmax.max.item(),
            "image_threshold": self.image_threshold.value.item(),
            "pixel_threshold": self.pixel_threshold.value.item(),
        }

        return out


class EnsemblePostProcessPipeline:
    """
    Pipeline used to perform various post-processing of ensemble predictions.

    Args:
        data: Class containing all tile predictions.
        joiner: Class used to join tiled data, already containing predictions.
    """

    def __init__(self, data: EnsemblePredictions, joiner: EnsemblePredictionJoiner):
        self.data = data
        self.joiner = joiner
        self.joiner.setup(self.data)

        self.steps: List[EnsemblePostProcess] = []

    def add_steps(self, steps: List[EnsemblePostProcess]) -> None:
        """
        Add list of sequential steps to pipeline.

        Args:
            steps: List containing blocks of pipeline.
        """
        self.steps = steps

    def execute(self) -> dict[str, Any]:
        """
        Execute the pipeline. For each batch go through all steps of pipeline.

        Returns:
            Dictionary containing results of `compute` function called for each step that has it.
        """

        for batch_index in tqdm(range(self.joiner.num_batches)):
            # first join every batch of tiles
            batch = self.joiner.join_tile_predictions(batch_index)
            # process the data through the pipeline.
            for step in tqdm(self.steps):
                batch = step.process(batch)

        # construct return dictionary with results of each block that has compute function.
        out = {}
        for step in tqdm(self.steps):
            if step.final_compute:
                out[step.name] = step.compute()

        return out
