"""Post-processor used in AnomalibTrainer.

This is responsible for setting up and computing thresholds.
"""
from __future__ import annotations

from typing import List

import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import Tensor

from anomalib.data.utils import boxes_to_anomaly_maps, boxes_to_masks, masks_to_boxes
from anomalib.models import AnomalyModule
from anomalib.post_processing import ThresholdMethod
from anomalib.utils.metrics import AnomalyScoreThreshold
from anomalib.utils.metrics.collection import AnomalibMetricCollection


class PostProcessor:
    """Post-processor used in AnomalibTrainer.

    Args:
        threshold_method (ThresholdMethod): Thresholding method to use.
        manual_image_threshold (Optional[float]): Image threshold in case manual threshold is used. Defaults to None.
        manual_pixel_threshold (Optional[float]) = Pixel threshold in case manual threshold is used. Defaults to None.
    """

    def __init__(
        self,
        threshold_method: ThresholdMethod,
        manual_image_threshold: float | None = None,
        manual_pixel_threshold: float | None = None,
    ):
        if threshold_method == ThresholdMethod.ADAPTIVE and all(
            i is not None for i in (manual_image_threshold, manual_pixel_threshold)
        ):
            raise ValueError(
                "When `threshold_method` is set to `adaptive`, `manual_image_threshold` and `manual_pixel_threshold` "
                "must not be set."
            )

        if threshold_method == ThresholdMethod.MANUAL and all(
            i is None for i in (manual_image_threshold, manual_pixel_threshold)
        ):
            raise ValueError(
                "When `threshold_method` is set to `manual`, `manual_image_threshold` and `manual_pixel_threshold` "
                "must be set."
            )

        self.threshold_method = threshold_method
        self.manual_image_threshold = manual_image_threshold
        self.manual_pixel_threshold = manual_pixel_threshold

    def setup(self, anomalib_module: AnomalyModule):
        """Assigns pixel and image thresholds to the model.

        This allows us to export the metrics along with the torch model.

        Args:
            anomalib_module (AnomalyModule): Anomaly module.
        """
        if not hasattr(anomalib_module, "pixel_threshold"):
            anomalib_module.pixel_threshold = AnomalyScoreThreshold().cpu()
        if not hasattr(anomalib_module, "image_threshold"):
            anomalib_module.image_threshold = AnomalyScoreThreshold().cpu()

        if self.threshold_method == ThresholdMethod.MANUAL:
            anomalib_module.pixel_threshold.value = torch.tensor(self.manual_pixel_threshold).cpu()
            anomalib_module.image_threshold.value = torch.tensor(self.manual_image_threshold).cpu()

    def _compute_adaptive_threshold(
        self, anomalib_module: AnomalyModule, outputs: EPOCH_OUTPUT | List[EPOCH_OUTPUT]
    ) -> None:
        anomalib_module.image_threshold.reset()
        anomalib_module.pixel_threshold.reset()
        self.update_metrics(anomalib_module.image_threshold, anomalib_module.pixel_threshold, outputs)
        anomalib_module.image_threshold.compute()
        if "mask" in outputs[0].keys() and "anomaly_maps" in outputs[0].keys():
            anomalib_module.pixel_threshold.compute()
        else:
            anomalib_module.pixel_threshold.value = anomalib_module.image_threshold.value

        anomalib_module.image_metrics.set_threshold(anomalib_module.image_threshold.value.item())
        anomalib_module.pixel_metrics.set_threshold(anomalib_module.pixel_threshold.value.item())

    def compute_threshold(self, anomalib_module: AnomalyModule, outputs: EPOCH_OUTPUT | List[EPOCH_OUTPUT]) -> None:
        """Computes adaptive threshold in case thresholding type is ADAPTIVE.

        Args:
            anomalib_module (AnomalyModule): Anomaly module.
            outputs (EPOCH_OUTPUT | List[EPOCH_OUTPUT]): Epoch end outputs.
        """
        if self.threshold_method == ThresholdMethod.ADAPTIVE:
            self._compute_adaptive_threshold(anomalib_module, outputs)

    @staticmethod
    def update_metrics(
        image_metric: AnomalibMetricCollection,
        pixel_metric: AnomalibMetricCollection,
        outputs: EPOCH_OUTPUT | List[EPOCH_OUTPUT] | STEP_OUTPUT,
    ) -> None:
        if isinstance(outputs, list):
            for output in outputs:
                PostProcessor.update_metrics(image_metric, pixel_metric, output)
        else:
            image_metric.cpu()
            image_metric.update(outputs["pred_scores"], outputs["label"].int())
            if "mask" in outputs.keys() and "anomaly_maps" in outputs.keys():
                pixel_metric.cpu()
                pixel_metric.update(outputs["anomaly_maps"], outputs["mask"].int())

    @staticmethod
    def post_process(outputs: STEP_OUTPUT) -> None:
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

    @classmethod
    def outputs_to_cpu(cls, output):
        """Move outputs to CPU."""
        if isinstance(output, dict):
            for key, value in output.items():
                output[key] = cls.outputs_to_cpu(value)
        elif isinstance(output, list):
            output = [cls.outputs_to_cpu(item) for item in output]
        elif isinstance(output, Tensor):
            output = output.cpu()
        return output

    @staticmethod
    def apply_thresholding(anomalib_module: AnomalyModule, outputs: STEP_OUTPUT):
        """Computes masks, box labels after applying thresholding."""
        # PostProcessor.post_process(outputs)
        if outputs is not None and isinstance(outputs, dict):
            outputs["pred_labels"] = outputs["pred_scores"] >= anomalib_module.image_threshold.value.cpu()
            if "anomaly_maps" in outputs.keys():
                outputs["pred_masks"] = outputs["anomaly_maps"] >= anomalib_module.pixel_threshold.value.cpu()
                if "pred_boxes" not in outputs.keys():
                    outputs["pred_boxes"], outputs["box_scores"] = masks_to_boxes(
                        outputs["pred_masks"], outputs["anomaly_maps"]
                    )
                    outputs["box_labels"] = [torch.ones(boxes.shape[0]) for boxes in outputs["pred_boxes"]]
            # apply thresholding to boxes
            if "box_scores" in outputs and "box_labels" not in outputs:
                # apply threshold to assign normal/anomalous label to boxes
                is_anomalous = [
                    scores > anomalib_module.pixel_threshold.value.cpu() for scores in outputs["box_scores"]
                ]
                outputs["box_labels"] = [labels.int() for labels in is_anomalous]
        return outputs
