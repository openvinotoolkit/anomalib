"""Post-processor used in AnomalibTrainer.

This is responsible for setting up and computing thresholds.
"""


import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from anomalib import trainer
from anomalib.data.utils import boxes_to_anomaly_maps, boxes_to_masks, masks_to_boxes


class PostProcessor:
    """Post-processor used in AnomalibTrainer."""

    def __init__(self, trainer: "trainer.AnomalibTrainer") -> None:
        self.trainer = trainer

    @staticmethod
    def apply_predictions(outputs: STEP_OUTPUT) -> None:
        """Computes prediction scores and prediction boxes."""
        PostProcessor._outputs_to_cpu(outputs)
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

    @staticmethod
    def _outputs_to_cpu(output):
        """Move outputs to CPU."""
        if isinstance(output, dict):
            for key, value in output.items():
                output[key] = PostProcessor._outputs_to_cpu(value)
        elif isinstance(output, list):
            output = [PostProcessor._outputs_to_cpu(item) for item in output]
        elif isinstance(output, Tensor):
            output = output.cpu()
        return output

    def apply_thresholding(self, outputs: STEP_OUTPUT):
        """Computes masks, box labels after applying thresholding."""
        if outputs is not None and isinstance(outputs, dict):
            outputs["pred_labels"] = outputs["pred_scores"] >= self.trainer.image_threshold.value.cpu()
            if "anomaly_maps" in outputs.keys():
                outputs["pred_masks"] = outputs["anomaly_maps"] >= self.trainer.pixel_threshold.value.cpu()
                if "pred_boxes" not in outputs.keys():
                    outputs["pred_boxes"], outputs["box_scores"] = masks_to_boxes(
                        outputs["pred_masks"], outputs["anomaly_maps"]
                    )
                    outputs["box_labels"] = [torch.ones(boxes.shape[0]) for boxes in outputs["pred_boxes"]]
            # apply thresholding to boxes
            if "box_scores" in outputs and "box_labels" not in outputs:
                # apply threshold to assign normal/anomalous label to boxes
                is_anomalous = [scores > self.trainer.pixel_threshold.value.cpu() for scores in outputs["box_scores"]]
                outputs["box_labels"] = [labels.int() for labels in is_anomalous]
        return outputs
