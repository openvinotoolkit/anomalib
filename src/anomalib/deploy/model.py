"""Torch model that includes pre and post processing for onnx and OpenVINO export."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import albumentations as A
import numpy as np
import torch
from albumentations.core.serialization import Serializable
from albumentations.pytorch import ToTensorV2
from kornia.contrib import connected_components
from omegaconf import DictConfig
from torch import Tensor, nn
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode

from anomalib.models import AnomalyModule
from anomalib.post_processing.normalization.cdf import normalize as normalize_cdf
from anomalib.post_processing.normalization.cdf import standardize
from anomalib.post_processing.normalization.min_max import normalize as normalize_min_max

ANOMALY_CLASS = torch.tensor(1, dtype=torch.uint8)
NORMAL_CLASS = torch.tensor(0, dtype=torch.uint8)
# A named tuple is required as the onnx export complains when using a dict or any other mutable data structure.
Result = namedtuple(
    "Result",
    ["anomaly_map", "pred_label", "pred_score", "pred_mask", "pred_boxes", "pred_boxes_per_image", "box_labels"],
)


class ExportModel(nn.Module):
    """Wraps the AnomalyModule with postprocessing method for export.
    This bypasses the need for exporting metadata with the model by adding normalization and thresholding to the model
    graph.

    Args:
        model (AnomalyModule): Anomaly model.
        input_size (tuple[int, int]): Input size of the model.
        metadata (dict | DictConfig | None): Metadata related to the model. Defaults to None.
    """

    def __init__(self, model: AnomalyModule, input_size: tuple[int, int], metadata: dict | DictConfig) -> None:
        super().__init__()
        self.model = model.model
        self.model.eval()
        self.device = model.device
        self.metadata = metadata
        self.input_size = input_size

        transform = A.from_dict(metadata["transform"])
        self.transform = self.albumentations_to_torch_vision(transform)  # convert transform for jit

    def albumentations_to_torch_vision(self, transforms: Serializable) -> Compose:
        """Since albumentations requires numpy arrays, we need to convert the transforms to torch transforms."""
        torch_transforms = []
        for transform in transforms:
            if isinstance(transform, ToTensorV2):
                continue  # Skip ToTensorV2 as it is already a tensor
            elif isinstance(transform, A.Resize):
                torch_transforms.append(
                    Resize(size=(transform.height, transform.width), interpolation=InterpolationMode.BILINEAR)
                )
            elif isinstance(transform, A.Normalize):
                torch_transforms.append(Normalize(mean=transform.mean, std=transform.std))
            elif isinstance(transform, A.CenterCrop):
                torch_transforms.append(CenterCrop(size=(transform.height, transform.width)))
            else:
                raise NotImplementedError(f"Transform {transform} not supported")

        return Compose(torch_transforms)

    def pre_process(self, image: Tensor) -> Tensor:
        """Pre process the input image by applying transformations.

        Args:
            image (np.ndarray): Input image

        Returns:
            Tensor: pre-processed image.
        """
        # convert image in range 0-255 to 0-1
        image = image.div(255.0)
        processed_image = self.transform(image)

        # if len(processed_image) == 3:
        #     processed_image = processed_image.unsqueeze(0)

        return processed_image

    def forward(self, image: torch.Tensor) -> Result:
        """Forward pass of the model.

        Args:
            image (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        image = self.pre_process(image)
        outputs = self.model(image)
        return self.post_process(predictions=outputs, metadata=self.metadata)

    @staticmethod
    def batch_mask_to_boxes(pred_mask: torch.Tensor, num_iterations: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts a batch of masks to bounding boxes.

        Args:
            pred_mask (Tensor): Batch of masks
            num_iterations (int, optional): Number of iterations for connected components. Defaults to 1000.

        Returns:
            Tensor: Bounding boxes
        """
        height, width = pred_mask.shape[-2:]
        pred_mask = pred_mask.view((-1, 1, height, width)).float()  # reshape to (B, 1, H, W) and cast to float

        batch_comps = connected_components(pred_mask, num_iterations=num_iterations).squeeze(1).int()
        B, H, W = batch_comps.shape

        labels = torch.unique(batch_comps)
        # Assign background (0) to -1
        labels[0] = -1.0

        masks = batch_comps.unsqueeze(1).repeat(1, labels.shape[0], 1, 1)
        masks = masks == labels.view(-1, 1, 1)

        # split the boxes later on based on this
        # since torch.count_nonzero is not supported in onnx, we use torch.sum with non equality
        # since mask is of shape B x N x H x W and N dimension has True values only in the mask of the corresponding
        # label in that image
        boxes_per_image = masks.view(B, labels.shape[0], -1).any(dim=-1).sum(dim=-1)

        bboxes = torch.full(
            (labels.shape[0], 4), fill_value=-1.0, device=pred_mask.device, dtype=torch.int64
        )  # N-1 x 4, since lables are unique and 0 is background

        # since the mask categories are mutually exclusive across batch, we can sum them to remove the batch dimension
        masks = torch.sum(masks, dim=0)  # N x H x W
        # Get the max and min values of the masks by multiplying each mask by a grid of its coordinates
        masks_ = masks.unsqueeze(1) * torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=0).to(
            pred_mask.device
        )  # N-1 x 2 x H x W
        masks_ = masks_.view(labels.size(0), 2, -1)  # N-1 x 2 x H*W
        # # assign bounding boxes
        bboxes[:, 2] = torch.max(masks_[:, 1], dim=1)[0]
        bboxes[:, 3] = torch.max(masks_[:, 0], dim=1)[0]
        masks_[masks_ == 0] = H + 1
        bboxes[:, 0] = torch.min(masks_[:, 1], dim=1)[0]
        bboxes[:, 1] = torch.min(masks_[:, 0], dim=1)[0]

        # remove the first bounding box since it is background
        bboxes = bboxes[1:]
        return bboxes, boxes_per_image

    def post_process(self, predictions: Tensor, metadata: dict | DictConfig | None = None) -> Result:
        """Post process the output predictions.

        Args:
            predictions (Tensor): Raw output predicted by the model.
            metadata (dict, optional): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.
                Defaults to None.

        Returns:
            dict[str, str | float | np.ndarray]: Post processed prediction results.
        """
        if metadata is None:
            metadata = self.metadata

        if isinstance(predictions, Tensor):
            anomaly_map = predictions
            pred_score = anomaly_map.reshape(-1).max()
            if len(anomaly_map.shape) <= 1:
                anomaly_map = None  # models like dfkde return a scalar
        else:
            # NOTE: Patchcore `forward`` returns heatmap and score.
            #   We need to add the following check to ensure the variables
            #   are properly assigned. Without this check, the code
            #   throws an error regarding type mismatch torch vs np.
            if isinstance(predictions[1], (Tensor)):
                anomaly_map, pred_score = predictions
                anomaly_map = anomaly_map
                pred_score = pred_score
            else:
                anomaly_map, pred_score = predictions
                pred_score = pred_score

        # Common practice in anomaly detection is to assign anomalous
        # label to the prediction if the prediction score is greater
        # than the image threshold.
        pred_label: str | None = None
        if "image_threshold" in metadata:
            pred_label = torch.where(
                pred_score >= metadata["image_threshold"],
                ANOMALY_CLASS,
                NORMAL_CLASS,
            )

        pred_mask: np.ndarray | None = None

        # TODO select based on task type
        if anomaly_map is not None:
            if "pixel_threshold" in metadata:
                pred_mask = (anomaly_map >= metadata["pixel_threshold"]).squeeze().type(torch.uint8)

            anomaly_map = anomaly_map.squeeze()
            anomaly_map, pred_score = self._normalize(
                anomaly_maps=anomaly_map, pred_scores=pred_score, metadata=metadata
            )

            image_height = self.input_size[0]
            image_width = self.input_size[1]
            anomaly_map = Resize((image_height, image_width))(anomaly_map.unsqueeze(0)).squeeze()

        if metadata["task"] == "detection" and pred_mask is not None:
            pred_boxes, pred_boxes_per_image = self.batch_mask_to_boxes(pred_mask)
            box_labels = torch.ones(pred_boxes.shape[0])
        else:
            pred_boxes = None
            box_labels = None
            pred_boxes_per_image = None

        return Result(
            anomaly_map=anomaly_map,
            pred_label=pred_label,
            pred_score=pred_score,
            pred_mask=pred_mask,
            pred_boxes=pred_boxes,
            pred_boxes_per_image=pred_boxes_per_image,  # currently unused in the inferencers
            box_labels=box_labels,
        )

    @staticmethod
    def _normalize(
        pred_scores: Tensor | np.float32,
        metadata: dict | DictConfig,
        anomaly_maps: Tensor | np.ndarray | None = None,
    ) -> tuple[Tensor | None, Tensor]:
        """Applies normalization and resizes the image.

        Args:
            pred_scores (Tensor | np.float32): Predicted anomaly score
            metadata (dict | DictConfig): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.
            anomaly_maps (Tensor | np.ndarray | None): Predicted raw anomaly map.

        Returns:
            tuple[np.ndarray | Tensor | None, float]: Post processed predictions that are ready to be
                visualized and predicted scores.
        """

        # min max normalization
        if "min" in metadata and "max" in metadata:
            if anomaly_maps is not None:
                anomaly_maps = normalize_min_max(
                    anomaly_maps,
                    metadata["pixel_threshold"],
                    metadata["min"],
                    metadata["max"],
                )
            pred_scores = normalize_min_max(
                pred_scores,
                metadata["image_threshold"],
                metadata["min"],
                metadata["max"],
            )

        # standardize pixel scores
        if "pixel_mean" in metadata.keys() and "pixel_std" in metadata.keys():
            if anomaly_maps is not None:
                anomaly_maps = standardize(
                    anomaly_maps, metadata["pixel_mean"], metadata["pixel_std"], center_at=metadata["image_mean"]
                )
                anomaly_maps = normalize_cdf(anomaly_maps, metadata["pixel_threshold"])

        # standardize image scores
        if "image_mean" in metadata.keys() and "image_std" in metadata.keys():
            pred_scores = standardize(pred_scores, metadata["image_mean"], metadata["image_std"])
            pred_scores = normalize_cdf(pred_scores, metadata["image_threshold"])

        return anomaly_maps, pred_scores
