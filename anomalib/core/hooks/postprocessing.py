"""Hooks for postprocessing the anomaly scores."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import Tensor
from torchmetrics import Metric

from anomalib.data.utils.boxes import boxes_to_anomaly_maps, boxes_to_masks, masks_to_boxes
from anomalib.models import AnomalyModule
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.metrics.min_max import MinMax

from .base import TrainerHooks


class PostProcessingHooks(TrainerHooks):
    def __init__(
        self,
        normalization_method: NormalizationMethod = NormalizationMethod.MIN_MAX,
        threshold_method: ThresholdMethod = ThresholdMethod.ADAPTIVE,
        normalization_metrics: Metric = MinMax(),
    ):
        self.normalization_method = normalization_method
        self.threshold_method = threshold_method
        self.normalization_metrics = normalization_metrics

    @staticmethod
    def _outputs_to_cpu(output):
        if isinstance(output, dict):
            for key, value in output.items():
                output[key] = PostProcessingHooks._outputs_to_cpu(value)
        elif isinstance(output, list):
            output = [PostProcessingHooks._outputs_to_cpu(item) for item in output]
        elif isinstance(output, Tensor):
            output = output.cpu()
        return output

    @staticmethod
    def _post_process(outputs: STEP_OUTPUT) -> None:
        """Compute labels based on model predictions."""
        if isinstance(outputs, dict):
            if "pred_scores" not in outputs and "anomaly_maps" in outputs:
                # infer image scores from anomaly maps
                outputs["pred_scores"] = (
                    outputs["anomaly_maps"].reshape(outputs["anomaly_maps"].shape[0], -1).max(dim=1).values
                )
            elif "pred_scores" not in outputs and "box_scores" in outputs:
                # infer image score from bbox confidence scores
                outputs["pred_scores"] = torch.zeros_like(outputs["label"]).float()
                for idx, (boxes, scores) in enumerate(zip(outputs["pred_boxes"], outputs["box_scores"])):
                    if boxes.numel():
                        outputs["pred_scores"][idx] = scores.max().item()

            if "pred_boxes" in outputs and "anomaly_maps" not in outputs:
                # create anomaly maps from bbox predictions for thresholding and evaluation
                image_size: tuple[int, int] = outputs["image"].shape[-2:]
                true_boxes: list[Tensor] = outputs["boxes"]
                pred_boxes: Tensor = outputs["pred_boxes"]
                box_scores: Tensor = outputs["box_scores"]

                outputs["anomaly_maps"] = boxes_to_anomaly_maps(pred_boxes, box_scores, image_size)
                outputs["mask"] = boxes_to_masks(true_boxes, image_size)

    def on_run_start(self, pl_module: AnomalyModule):
        """Setup thresholding method and the thresholds.

        Adds these attributes to the lightning module
        """
        setattr(pl_module, "threshold_method", self.threshold_method)
        setattr(pl_module, "normalization_metrics", self.normalization_metrics)

    def test_step(self, pl_module: AnomalyModule, outputs: STEP_OUTPUT) -> None:
        self.validation_step(pl_module, outputs)
        if outputs is not None and isinstance(outputs, dict):
            outputs["pred_labels"] = outputs["pred_scores"] >= pl_module.image_threshold.value
            if "anomaly_maps" in outputs.keys():
                outputs["pred_masks"] = outputs["anomaly_maps"] >= pl_module.pixel_threshold.value
                if "pred_boxes" not in outputs.keys():
                    outputs["pred_boxes"], outputs["box_scores"] = masks_to_boxes(
                        outputs["pred_masks"], outputs["anomaly_maps"]
                    )
                    outputs["box_labels"] = [torch.ones(boxes.shape[0]) for boxes in outputs["pred_boxes"]]
            # apply thresholding to boxes
            if "box_scores" in outputs and "box_labels" not in outputs:
                # apply threshold to assign normal/anomalous label to boxes
                is_anomalous = [scores > pl_module.pixel_threshold.value for scores in outputs["box_scores"]]
                outputs["box_labels"] = [labels.int() for labels in is_anomalous]

    def test_epoch_end(self, pl_module: AnomalyModule, outputs: EPOCH_OUTPUT) -> None:
        pl_module._collect_outputs(pl_module.image_metrics, pl_module.pixel_metrics, outputs)

    def predict_step(self, lightning_module: AnomalyModule, outputs: list[STEP_OUTPUT]) -> None:
        """Predicted outputs.

        Args:
            outputs (List[Tensor] | List[dict[str, Tensor]]): Input batch.
        """
        for output in outputs:
            self._post_process(output)
            self.test_step(lightning_module, output)

    def validation_step(self, pl_module: AnomalyModule, outputs: STEP_OUTPUT):
        self._outputs_to_cpu(outputs)
        self._post_process(outputs)

    def validation_batch_end(self, pl_module: AnomalyModule, outputs: STEP_OUTPUT):
        if isinstance(outputs, dict):
            if "anomaly_maps" in outputs:
                pl_module.normalization_metrics(outputs["anomaly_maps"])
            elif "box_scores" in outputs:
                pl_module.normalization_metrics(torch.cat(outputs["box_scores"]))
            elif "pred_scores" in outputs:
                pl_module.normalization_metrics(outputs["pred_scores"])
            else:
                raise ValueError(
                    "No values found for normalization, provide anomaly maps, bbox scores, or image scores"
                )

    def validation_epoch_end(self, pl_module: AnomalyModule, outputs: EPOCH_OUTPUT):
        if pl_module.threshold_method == ThresholdMethod.ADAPTIVE:
            pl_module._compute_adaptive_threshold(outputs)
        pl_module._collect_outputs(pl_module.image_metrics, pl_module.pixel_metrics, outputs)
