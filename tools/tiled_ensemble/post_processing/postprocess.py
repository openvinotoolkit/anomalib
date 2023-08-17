"""Classes and functions used for various postprocessing of ensemble predictions."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from enum import Enum
from typing import Any

import torch
from tools.tiled_ensemble.ensemble_tiler import EnsembleTiler
from tools.tiled_ensemble.predictions import EnsemblePredictionJoiner, EnsemblePredictions
from torch import Tensor
from tqdm import tqdm

from anomalib.data.utils import masks_to_boxes
from anomalib.models.components import GaussianBlur2d
from anomalib.post_processing.normalization.min_max import normalize
from anomalib.utils.metrics import AnomalyScoreThreshold, MinMax


class ThresholdStage(str, Enum):
    """
    Enum signaling at which stage the thresholding is applied.

    In case of individual_tile, thresholding is applied for each tile location separately.
    In case of joined_image, thresholding is applied at the end when images are joined back together.
    """

    INDIVIDUAL_TILE = "individual_tile"
    JOINED_IMAGE = "joined_image"


class NormalizationStage(str, Enum):
    """
    Enum signaling at which stage the normalization is done.

    In case of individual_tile, tiles are normalized for each tile position separately.
    In case of joined_image, normalization is done at the end when images are joined back together.
    In case of none, output is not normalized.
    """

    INDIVIDUAL_TILE = "individual_tile"
    JOINED_IMAGE = "joined_image"
    NONE = "none"


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
    """Smooth the regions where tiles join in ensemble.

    It is recommended that thresholding is done again after smoothing to obtain new masks.

    Args:
        width_factor (float):  Factor multiplied by tile dimension to get the region around join which will be smoothed.
        filter_sigma (float): Sigma of filter used for smoothing the joins.
        tiler (EnsembleTiler): Tiler object used to get tile dimension data.

    Example:
        >>> from tools.tiled_ensemble.ensemble_tiler import EnsembleTiler
        >>> from tools.tiled_ensemble.predictions.basic_joiner import BasicPredictionJoiner
        >>>
        >>> tiler = EnsembleTiler(tile_size=256, stride=128, image_size=512)
        >>> joiner = BasicPredictionJoiner(tiler)
        >>> pipeline = EnsemblePostProcessPipeline(joiner)
        >>>
        >>> # This will smooth 10% on each side of join with gaussian filter that has sigma=2
        >>> smooth = SmoothJoins(width_factor=0.1, filter_sigma=2, tiler=tiler)
        >>>
        >>> # this block can then be added to pipeline
        >>> pipeline.add_steps([smooth])
    """

    def __init__(self, width_factor: float, filter_sigma: float, tiler: EnsembleTiler) -> None:
        super().__init__(final_compute=False, name="smooth_joins")
        # offset in pixels of region around tile join that will be smoothed
        self.height_offset = int(tiler.tile_size_h * width_factor)
        self.width_offset = int(tiler.tile_size_w * width_factor)
        self.tiler = tiler

        self.join_mask = self.prepare_join_mask()

        self.blur = GaussianBlur2d(sigma=filter_sigma)

    def prepare_join_mask(self) -> Tensor:
        """
        Prepare boolean mask of regions around the part where tiles join in ensemble.

        Returns:
            Tensor representation of boolean mask where filtered joins should be used.
        """
        img_h, img_w = self.tiler.image_size
        stride_h, stride_w = self.tiler.stride_h, self.tiler.stride_w

        mask = torch.zeros(img_h, img_w, dtype=torch.bool)

        # prepare mask strip on vertical joins
        curr_w = stride_w
        while curr_w < img_w:
            start_i = curr_w - self.width_offset
            end_i = curr_w + self.width_offset
            mask[:, start_i:end_i] = 1
            curr_w += stride_w

        # prepare mask strip on horizontal joins
        curr_h = stride_h
        while curr_h < img_h:
            start_i = curr_h - self.height_offset
            end_i = curr_h + self.height_offset
            mask[start_i:end_i, :] = True
            curr_h += stride_h

        return mask

    def process(self, data: dict) -> dict:
        """
        Smooth the parts where tiles join in anomaly maps.

        Args:
            data: Predictions from ensemble pipeline.

        Returns:
            Predictions where anomaly maps are smoothed on tile joins.
        """
        smoothed = self.blur(data["anomaly_maps"])
        data["anomaly_maps"][:, :, self.join_mask] = smoothed[:, :, self.join_mask]

        return data


class MinMaxNormalize(EnsemblePostProcess):
    """
    Normalize images using min max normalization.

    Args:
        stats: dictionary containing statistics used for normalization (min, max, image threshold, pixel threshold).

    Example:
        >>> from tools.tiled_ensemble.ensemble_tiler import EnsembleTiler
        >>> from tools.tiled_ensemble.predictions.basic_joiner import BasicPredictionJoiner
        >>> from tools.tiled_ensemble.post_processing.pipelines import get_stats
        >>>
        >>> tiler = EnsembleTiler(tile_size=256, stride=128, image_size=512)
        >>> joiner = BasicPredictionJoiner(tiler)
        >>> pipeline = EnsemblePostProcessPipeline(joiner)
        >>>
        >>> # get statistics on validation data
        >>> stats = get_stats(...)
        >>> # based on stats (min, max & thresholds of validation data) this block will normalize input data
        >>> normalization = MinMaxNormalize(stats)
        >>>
        >>> # this block can then be added to pipeline
        >>> pipeline.add_steps([normalization])
    """

    def __init__(self, stats: dict[str, float]) -> None:
        super().__init__(final_compute=False, name="minmax_normalize")

        self.image_threshold = stats["image_threshold"]
        self.pixel_threshold = stats["pixel_threshold"]
        self.min_val = stats["min"]
        self.max_val = stats["max"]

    def process(self, data: dict) -> dict:
        """
        Normalize predictions using minmax normalization.

        Args:
            data: Predictions from ensemble pipeline.

        Returns:
            Normalized predictions.
        """
        data["pred_scores"] = normalize(data["pred_scores"], self.image_threshold, self.min_val, self.max_val)
        if "anomaly_maps" in data:
            data["anomaly_maps"] = normalize(data["anomaly_maps"], self.pixel_threshold, self.min_val, self.max_val)
        if "box_scores" in data:
            data["box_scores"] = [
                normalize(scores, self.pixel_threshold, self.min_val, self.max_val) for scores in data["box_scores"]
            ]

        return data


class Threshold(EnsemblePostProcess):
    """
    Threshold the predictions using provided thresholds.

    Args:
        image_threshold: Threshold used for image-level thresholding.
        pixel_threshold: Threshold used for pixel-level thresholding.

    Example:
        >>> from tools.tiled_ensemble.ensemble_tiler import EnsembleTiler
        >>> from tools.tiled_ensemble.predictions.basic_joiner import BasicPredictionJoiner
        >>> from tools.tiled_ensemble.post_processing.pipelines import get_stats
        >>>
        >>> tiler = EnsembleTiler(tile_size=256, stride=128, image_size=512)
        >>> joiner = BasicPredictionJoiner(tiler)
        >>> pipeline = EnsemblePostProcessPipeline(joiner)
        >>>
        >>> # get statistics on validation data
        >>> stats = get_stats(...)
        >>> # based on stats (image & pixel threshold) this block will threshold the data to get labels, masks and boxes
        >>> threshold = Threshold(stats["image_threshold"], stats["pixel_threshold"])
        >>>
        >>> # this block can then be added to pipeline
        >>> pipeline.add_steps([threshold])
    """

    def __init__(self, image_threshold: float, pixel_threshold: float) -> None:
        super().__init__(final_compute=False, name="threshold")
        self.image_threshold = image_threshold
        self.pixel_threshold = pixel_threshold

    def process(self, data: dict) -> dict:
        """
        Threshold all prediction data: labels, pixels and boxes.

        Args:
            data: Predictions from ensemble pipeline.

        Returns:
            Predictions with threshold applied.
        """
        data["pred_labels"] = data["pred_scores"] >= self.image_threshold
        if "anomaly_maps" in data.keys():
            data["pred_masks"] = data["anomaly_maps"] >= self.pixel_threshold

            # also make boxes from predicted masks
            data["pred_boxes"], data["box_scores"] = masks_to_boxes(data["pred_masks"], data["anomaly_maps"])
            data["box_labels"] = [torch.ones(boxes.shape[0]) for boxes in data["pred_boxes"]]
        # apply thresholding to boxes
        if "box_scores" in data and "box_labels" not in data:
            # apply threshold to assign normal/anomalous label to boxes
            is_anomalous = [scores > self.pixel_threshold for scores in data["box_scores"]]
            data["box_labels"] = [labels.int() for labels in is_anomalous]

        return data


class PostProcessStats(EnsemblePostProcess):
    """Class used to obtain threshold and normalization statistics: (min, max, image threshold, pixel threshold).

    Example:
        >>> from tools.tiled_ensemble.ensemble_tiler import EnsembleTiler
        >>> from tools.tiled_ensemble.predictions.basic_joiner import BasicPredictionJoiner
        >>>
        >>> tiler = EnsembleTiler(tile_size=256, stride=128, image_size=512)
        >>> joiner = BasicPredictionJoiner(tiler)
        >>> pipeline = EnsemblePostProcessPipeline(joiner)
        >>>
        >>> # this block calculates min, max and image & pixel threshold on validation data.
        >>> stats_block = PostProcessStats()
        >>>
        >>> # this block can then be added to pipeline
        >>> pipeline.add_steps([stats_block])
    """

    def __init__(self) -> None:
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
        joiner: Class used to join tiled data, already containing predictions.

    Example:
        >>> from tools.tiled_ensemble.ensemble_tiler import EnsembleTiler
        >>> from tools.tiled_ensemble.predictions.basic_joiner import BasicPredictionJoiner
        >>> from tools.tiled_ensemble.predictions.prediction_data import BasicEnsemblePredictions
        >>> from tools.tiled_ensemble.post_processing.metrics import EnsembleMetrics

        >>> tiler = EnsembleTiler(tile_size=256, stride=128, image_size=512)
        >>> joiner = BasicPredictionJoiner(tiler)
        >>> data = BasicEnsemblePredictions()
        >>> # ... data is then filed with predictions from ensemble
        >>>
        >>> # make instance of pipeline with joiner
        >>> pipeline = EnsemblePostProcessPipeline(joiner)
        >>>
        >>> # steps can then be added in order which they will be executed in
        >>> pipeline.add_steps([SmoothJoins(...), MinMaxNormalize(...), Threshold(...), EnsembleMetrics(...)])
        >>>
        >>> # pipeline is then execute on given data
        >>> pipe_out = pipeline.execute(data)
        >>> pipe_out
        {'metrics': {'image_F1Score': 0.42,
                     'image_AUROC': 0.42,
                     'pixel_F1Score': 0.42,
                     'pixel_AUROC': 0.42,
                     'pixel_AUPRO': 0.42}}
    """

    def __init__(self, joiner: EnsemblePredictionJoiner) -> None:
        self.joiner = joiner

        self.steps: list[EnsemblePostProcess] = []

    def add_steps(self, steps: list[EnsemblePostProcess]) -> None:
        """
        Add list of sequential steps to pipeline.

        Args:
            steps: List containing blocks of pipeline.
        """
        self.steps = steps

    @torch.inference_mode()
    def execute(self, data: EnsemblePredictions) -> dict[str, Any]:
        """
        Execute the pipeline. For each batch go through all steps of pipeline.

        Args:
            data: Class containing all tile predictions.

        Returns:
            Dictionary containing results of `compute` function called for each step that has it.
        """
        # setup joiner to process given tiled data
        self.joiner.setup(data)

        for batch_index in tqdm(range(self.joiner.num_batches)):
            # first join every batch of tiles
            batch = self.joiner.join_tile_predictions(batch_index)

            # process the data through the pipeline.
            for step in self.steps:
                batch = step.process(batch)

        # construct return dictionary with results of each block that has compute function.
        out = {}
        for step in tqdm(self.steps):
            if step.final_compute:
                out[step.name] = step.compute()

        return out
