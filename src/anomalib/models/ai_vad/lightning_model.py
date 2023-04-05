"""Attribute-based Representations for Accurate and Interpretable Video Anomaly Detection.

Paper https://arxiv.org/pdf/2212.00789.pdf
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from typing import List
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from anomalib.models.components import AnomalyModule
from anomalib.models.ai_vad.torch_model import AiVadModel
from torchvision.transforms.functional import gaussian_blur

logger = logging.getLogger(__name__)

__all__ = ["AiVad", "AiVadLightning"]


class AiVad(AnomalyModule):
    """PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.

    Args:
        layers (list[str]): Layers to extract features from the backbone CNN
        input_size (tuple[int, int]): Size of the model input.
        backbone (str): Backbone CNN network
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        n_features (int, optional): Number of features to retain in the dimension reduction step.
                                Default values from the paper are available for: resnet18 (100), wide_resnet50_2 (550).
    """

    def __init__(
        self,
        box_score_thresh: float = 0.8,
        n_velocity_bins: int = 8,
        use_velocity_features: bool = True,
        use_pose_features: bool = True,
        use_appearance_features: bool = True,
        n_components_velocity: int = 5,
        n_neighbors_pose: int = 1,
        n_neighbors_appearance: int = 1,
    ) -> None:
        super().__init__()

        self.model = AiVadModel(
            box_score_thresh=box_score_thresh,
            n_velocity_bins=n_velocity_bins,
            use_velocity_features=use_velocity_features,
            use_pose_features=use_pose_features,
            use_appearance_features=use_appearance_features,
            n_components_velocity=n_components_velocity,
            n_neighbors_pose=n_neighbors_pose,
            n_neighbors_appearance=n_neighbors_appearance,
        )

    @staticmethod
    def configure_optimizers() -> None:  # pylint: disable=arguments-differ
        return None

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> None:
        features_per_batch = self.model(batch["image"])

        for features, video_path in zip(features_per_batch, batch["video_path"]):
            self.model.density_estimator.update(features, video_path)

    def on_validation_start(self) -> None:
        self.model.density_estimator.fit()

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        boxes, anomaly_scores = self.model(batch["image"])
        batch["pred_boxes"] = [box.int() for box in boxes]
        batch["box_scores"] = [score.to(self.device) for score in anomaly_scores]

        # TODO: this should be handled by video dataset
        batch["boxes"] = [boxes[-1] for boxes in batch["boxes"]]
        batch["mask"] = batch["mask"][:, -1, ...]
        batch["image"] = batch["image"][:, -1, ...]
        batch["original_image"] = batch["original_image"][:, -1, ...]
        batch["label"] = batch["label"][:, -1]
        batch["frames"] = batch["frames"][:, -1]

        return batch

    def validation_epoch_end(self, outputs) -> None:
        # smoothen frame-level predictions using 1d Gaussian filter
        sigma = 3
        kernel_size = 5

        # collect in dict
        anomaly_score_dict = {}
        for batch in outputs:
            # group by video id
            for pred_score, vid_id, frames in zip(batch["pred_scores"], batch["video_path"], batch["frames"]):
                if vid_id not in anomaly_score_dict:
                    anomaly_score_dict[vid_id] = {"pred_scores": [], "frames": []}
                anomaly_score_dict[vid_id]["pred_scores"].append(pred_score)
                anomaly_score_dict[vid_id]["frames"].append(frames)

        for vid_id, score_dict in anomaly_score_dict.items():
            score_dict["pred_scores"] = torch.stack(score_dict["pred_scores"])
            scores = score_dict["pred_scores"]
            # TODO: re-order based on frames before filtering
            if len(scores) >= kernel_size - 2:  # minimum length depends on kernel size
                scores = gaussian_blur(scores.reshape(1, 1, -1), kernel_size=[kernel_size, 1], sigma=sigma)
            score_dict["filtered_scores"] = scores.reshape(-1)
            score_dict["frames"] = torch.stack(score_dict["frames"])

        for batch in outputs:
            new_batch_scores = torch.zeros_like(batch["pred_scores"])
            for idx, (pred_score, vid_id, frame) in enumerate(
                zip(batch["pred_scores"], batch["video_path"], batch["frames"])
            ):
                filtered_frames = anomaly_score_dict[vid_id]["frames"]
                filtered_scores = anomaly_score_dict[vid_id]["filtered_scores"]

                new_batch_scores[idx] = filtered_scores[filtered_frames == frame]

            batch["pred_scores"] = new_batch_scores

        return outputs


class AiVadLightning(AiVad):
    """PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(
            box_score_thresh=hparams.model.box_score_thresh,
            n_velocity_bins=hparams.model.n_velocity_bins,
            use_velocity_features=hparams.model.use_velocity_features,
            use_pose_features=hparams.model.use_pose_features,
            use_appearance_features=hparams.model.use_appearance_features,
            n_components_velocity=hparams.model.n_components_velocity,
            n_neighbors_pose=hparams.model.n_neighbors_pose,
            n_neighbors_appearance=hparams.model.n_neighbors_appearance,
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)
