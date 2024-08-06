"""Callback that attaches necessary pre/post-processing to the model."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from lightning import Callback
from lightning.pytorch import Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib.data.utils import boxes_to_anomaly_maps, boxes_to_masks, masks_to_boxes
from anomalib.dataclasses import Batch
from anomalib.models import AnomalyModule


class _PostProcessorCallback(Callback):
    """Applies post-processing to the model outputs.

    Note: This callback is set within the Engine.
    """

    def __init__(self) -> None:
        super().__init__()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: Batch,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del batch, batch_idx, dataloader_idx  # Unused arguments.

        if outputs is not None:
            self.post_process(trainer, pl_module, outputs)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: Batch,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del batch, batch_idx, dataloader_idx  # Unused arguments.

        if outputs is not None:
            self.post_process(trainer, pl_module, outputs)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: Batch,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del batch, batch_idx, dataloader_idx  # Unused arguments.

        if outputs is not None:
            self.post_process(trainer, pl_module, outputs)

    def post_process(self, trainer: Trainer, pl_module: AnomalyModule, outputs: STEP_OUTPUT) -> None:
        if isinstance(outputs, Batch):
            self._post_process(outputs)
            if trainer.predicting or trainer.testing:
                self._compute_scores_and_labels(pl_module, outputs)

    @staticmethod
    def _compute_scores_and_labels(
        pl_module: AnomalyModule,
        outputs: Batch,
    ) -> None:
        if outputs.pred_score is not None:
            outputs.pred_label = outputs.pred_score >= pl_module.image_threshold.value
        if outputs.anomaly_map is not None:
            outputs.pred_mask = outputs.anomaly_map >= pl_module.pixel_threshold.value
            if outputs.pred_boxes is None:
                outputs.pred_boxes, outputs.box_scores = masks_to_boxes(
                    outputs.pred_mask,
                    outputs.anomaly_map,
                )
                outputs.box_labels = [torch.ones(boxes.shape[0]) for boxes in outputs.pred_boxes]
        # apply thresholding to boxes
        if outputs.box_scores is not None and outputs.box_labels is None:
            # apply threshold to assign normal/anomalous label to boxes
            is_anomalous = [scores > pl_module.pixel_threshold.value for scores in outputs.box_scores]
            outputs.box_labels = [labels.int() for labels in is_anomalous]

    @staticmethod
    def _post_process(outputs: Batch) -> None:
        """Compute labels based on model predictions."""
        if isinstance(outputs, Batch):
            if outputs.pred_score is None and outputs.anomaly_map is not None:
                # infer image scores from anomaly maps
                outputs.pred_score = outputs.anomaly_map.reshape(outputs.anomaly_map.shape[0], -1).max(dim=1)[0]

            if outputs.pred_boxes is not None and outputs.anomaly_map is None:
                # create anomaly maps from bbox predictions for thresholding and evaluation
                assert outputs.image is not None
                image_size: tuple[int, int] = outputs.image.shape[-2:]
                pred_boxes: torch.Tensor = outputs.pred_boxes
                box_scores: torch.Tensor = outputs.box_scores

                outputs.anomaly_map = boxes_to_anomaly_maps(pred_boxes, box_scores, image_size)

                if outputs.gt_boxes is not None:
                    true_boxes: list[torch.Tensor] = outputs.gt_boxes
                    outputs.gt_mask = boxes_to_masks(true_boxes, image_size)
