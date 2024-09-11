"""Validate torch image data."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np
import torch
from torchvision.transforms.v2.functional import to_dtype_image
from torchvision.tv_tensors import Image, Mask

from anomalib.data.validators.path import validate_path


class ImageValidator:
    """Validate torch.Tensor data for images."""

    @staticmethod
    def validate_image(image: torch.Tensor) -> Image:
        """Validate and convert the input image tensor.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            Image: Validated and converted image.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the image shape or number of channels is invalid.

        Examples:
            >>> import torch
            >>> from anomalib.dataclasses.validators import ImageValidator
            >>> tensor = torch.rand(3, 224, 224)
            >>> validated_image = ImageValidator.validate_image(tensor)
            >>> isinstance(validated_image, Image)
            True
            >>> validated_image.shape
            torch.Size([1, 3, 224, 224])
        """
        if not isinstance(image, torch.Tensor):
            msg = f"Image must be a torch.Tensor, got {type(image)}."
            raise TypeError(msg)

        if image.ndim not in {3, 4}:
            msg = f"Image must have shape [C, H, W] or [N, C, H, W], got shape {image.shape}."
            raise ValueError(msg)

        if image.ndim == 3:
            image = image.unsqueeze(0)  # add batch dimension

        if image.shape[1] not in {1, 3, 4}:
            msg = f"Invalid number of channels: {image.shape[1]}. Expected 1, 3, or 4."
            raise ValueError(msg)

        return Image(to_dtype_image(image, dtype=torch.float32, scale=True))

    @staticmethod
    def validate_gt_label(label: int | torch.Tensor | None) -> torch.Tensor | None:
        """Validate the ground truth label.

        Args:
            label (int | torch.Tensor | None): Input ground truth label.

        Returns:
            torch.Tensor | None: Validated ground truth label as a boolean tensor, or None.

        Raises:
            TypeError: If the input is neither an integer nor a torch.Tensor.
            ValueError: If the label shape or dtype is invalid.

        Examples:
            >>> import torch
            >>> from anomalib.dataclasses.validators import ImageValidator
            >>> label_int = 1
            >>> validated_label = ImageValidator.validate_gt_label(label_int)
            >>> validated_label
            tensor(True)
            >>> label_tensor = torch.tensor(0)
            >>> validated_label = ImageValidator.validate_gt_label(label_tensor)
            >>> validated_label
            tensor(False)
        """
        if label is None:
            return None
        if isinstance(label, int):
            return torch.tensor(label, dtype=torch.bool)
        if isinstance(label, torch.Tensor):
            if label.ndim != 0:
                msg = f"Ground truth label must be a scalar, got shape {label.shape}."
                raise ValueError(msg)
            if torch.is_floating_point(label):
                msg = f"Ground truth label must be boolean or integer, got {label.dtype}."
                raise ValueError(msg)
            return label.bool()
        msg = f"Ground truth label must be an integer or a torch.Tensor, got {type(label)}."
        raise TypeError(msg)

    @staticmethod
    def validate_gt_mask(mask: torch.Tensor | None) -> Mask | None:
        """Validate the ground truth mask.

        Args:
            mask (torch.Tensor | None): Input ground truth mask.

        Returns:
            Mask | None: Validated ground truth mask, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the mask shape is invalid.

        Examples:
            >>> import torch
            >>> from anomalib.dataclasses.validators import ImageValidator
            >>> mask = torch.randint(0, 2, (1, 224, 224))
            >>> validated_mask = ImageValidator.validate_gt_mask(mask)
            >>> isinstance(validated_mask, Mask)
            True
            >>> validated_mask.shape
            torch.Size([224, 224])
        """
        if mask is None:
            return None
        if not isinstance(mask, torch.Tensor):
            msg = f"Ground truth mask must be a torch.Tensor, got {type(mask)}."
            raise TypeError(msg)
        if mask.ndim not in {2, 3}:
            msg = f"Ground truth mask must have shape [H, W] or [1, H, W], got shape {mask.shape}."
            raise ValueError(msg)
        if mask.ndim == 3:
            if mask.shape[0] != 1:
                msg = f"Ground truth mask must have 1 channel, got {mask.shape[0]}."
                raise ValueError(msg)
            mask = mask.squeeze(0)
        return Mask(mask, dtype=torch.bool)

    @staticmethod
    def validate_anomaly_map(anomaly_map: torch.Tensor | None) -> Mask | None:
        """Validate the anomaly map.

        Args:
            anomaly_map (torch.Tensor | None): Input anomaly map.

        Returns:
            Mask | None: Validated anomaly map as a Mask, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the anomaly map shape is invalid.

        Examples:
            >>> import torch
            >>> from anomalib.dataclasses.validators import ImageValidator
            >>> anomaly_map = torch.rand(1, 224, 224)
            >>> validated_map = ImageValidator.validate_anomaly_map(anomaly_map)
            >>> isinstance(validated_map, Mask)
            True
            >>> validated_map.shape
            torch.Size([224, 224])
        """
        if anomaly_map is None:
            return None
        if not isinstance(anomaly_map, torch.Tensor):
            msg = f"Anomaly map must be a torch.Tensor, got {type(anomaly_map)}."
            raise TypeError(msg)
        if anomaly_map.ndim not in {2, 3}:
            msg = f"Anomaly map must have shape [H, W] or [1, H, W], got shape {anomaly_map.shape}."
            raise ValueError(msg)
        if anomaly_map.ndim == 3:
            if anomaly_map.shape[0] != 1:
                msg = f"Anomaly map must have 1 channel, got {anomaly_map.shape[0]}."
                raise ValueError(msg)
            anomaly_map = anomaly_map.squeeze(0)
        return Mask(anomaly_map)

    @staticmethod
    def validate_image_path(image_path: str | None) -> str | None:
        """Validate the image path.

        Args:
            image_path (str | None): Input image path.

        Returns:
            str | None: Validated image path, or None.

        Examples:
            >>> from anomalib.dataclasses.validators import ImageValidator
            >>> path = "/path/to/image.jpg"
            >>> validated_path = ImageValidator.validate_image_path(path)
            >>> validated_path == path
            True
        """
        return validate_path(image_path) if image_path else None

    @staticmethod
    def validate_mask_path(mask_path: str | None) -> str | None:
        """Validate the mask path.

        Args:
            mask_path (str | None): Input mask path.

        Returns:
            str | None: Validated mask path, or None.

        Examples:
            >>> from anomalib.dataclasses.validators import ImageValidator
            >>> path = "/path/to/mask.png"
            >>> validated_path = ImageValidator.validate_mask_path(path)
            >>> validated_path == path
            True
        """
        return validate_path(mask_path) if mask_path else None

    @staticmethod
    def validate_pred_score(pred_score: torch.Tensor | float | None) -> torch.Tensor | None:
        """Validate the prediction score.

        Args:
            pred_score (torch.Tensor | float | None): Input prediction score.

        Returns:
            torch.Tensor | None: Validated prediction score as a float32 tensor, or None.

        Raises:
            TypeError: If the input is neither a float, torch.Tensor, nor None.
            ValueError: If the prediction score is not a scalar.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import ImageValidator
            >>> score = 0.8
            >>> validated_score = ImageValidator.validate_pred_score(score)
            >>> validated_score
            tensor(0.8000)
            >>> score_tensor = torch.tensor(0.7)
            >>> validated_score = ImageValidator.validate_pred_score(score_tensor)
            >>> validated_score
            tensor(0.7000)
            >>> validated_score = ImageValidator.validate_pred_score(None)
            >>> validated_score is None
            True
        """
        if pred_score is None:
            return None
        if isinstance(pred_score, float):
            pred_score = torch.tensor(pred_score)
        if not isinstance(pred_score, torch.Tensor):
            msg = f"Prediction score must be a torch.Tensor, float, or None, got {type(pred_score)}."
            raise TypeError(msg)
        pred_score = pred_score.squeeze()
        if pred_score.ndim != 0:
            msg = f"Prediction score must be a scalar, got shape {pred_score.shape}."
            raise ValueError(msg)
        return pred_score.to(torch.float32)

    @staticmethod
    def validate_pred_mask(pred_mask: torch.Tensor | None) -> Mask | None:
        """Validate the prediction mask.

        Args:
            pred_mask (torch.Tensor | None): Input prediction mask.

        Returns:
            Mask | None: Validated prediction mask, or None.

        Examples:
            >>> import torch
            >>> from anomalib.dataclasses.validators import ImageValidator
            >>> mask = torch.randint(0, 2, (1, 224, 224))
            >>> validated_mask = ImageValidator.validate_pred_mask(mask)
            >>> isinstance(validated_mask, Mask)
            True
            >>> validated_mask.shape
            torch.Size([224, 224])
        """
        return ImageValidator.validate_gt_mask(pred_mask)  # We can reuse the gt_mask validation

    @staticmethod
    def validate_pred_label(pred_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction label.

        Args:
            pred_label (torch.Tensor | None): Input prediction label.

        Returns:
            torch.Tensor | None: Validated prediction label as a boolean tensor, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the prediction label is not a scalar.

        Examples:
            >>> import torch
            >>> from anomalib.dataclasses.validators import ImageValidator
            >>> label = torch.tensor(1)
            >>> validated_label = ImageValidator.validate_pred_label(label)
            >>> validated_label
            tensor(True)
        """
        if pred_label is None:
            return None
        if not isinstance(pred_label, torch.Tensor):
            msg = f"Predicted label must be a torch.Tensor, got {type(pred_label)}."
            raise TypeError(msg)
        pred_label = pred_label.squeeze()
        if pred_label.ndim != 0:
            msg = f"Predicted label must be a scalar, got shape {pred_label.shape}."
            raise ValueError(msg)
        return pred_label.to(torch.bool)


class ImageBatchValidator:
    """Validate torch.Tensor data for batches of images."""

    @staticmethod
    def validate_image(images: torch.Tensor) -> Image:
        """Validate and convert a batch of image tensors.

        Args:
            images (torch.Tensor): Input batch of image tensors.

        Returns:
            Image: Validated and converted batch of images.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the image shape or number of channels is invalid.

        Examples:
            >>> import torch
            >>> tensor = torch.rand(4, 3, 224, 224)  # Batch of 4 images
            >>> validated_images = ImageBatchValidator.validate_image(tensor)
            >>> isinstance(validated_images, Image)
            True
            >>> validated_images.shape
            torch.Size([4, 3, 224, 224])
        """
        if not isinstance(images, torch.Tensor):
            msg = f"Images must be a torch.Tensor, got {type(images)}."
            raise TypeError(msg)

        if images.ndim != 4:
            msg = f"Images must have shape [N, C, H, W], got shape {images.shape}."
            raise ValueError(msg)

        if images.shape[1] not in {1, 3, 4}:
            msg = f"Invalid number of channels: {images.shape[1]}. Expected 1, 3, or 4."
            raise ValueError(msg)

        return Image(to_dtype_image(images, dtype=torch.float32, scale=True))

    @staticmethod
    def validate_gt_label(labels: torch.Tensor | Sequence[int] | None) -> torch.Tensor | None:
        """Validate the ground truth labels for a batch.

        Args:
            labels (torch.Tensor | list[int] | None): Input ground truth labels.

        Returns:
            torch.Tensor | None: Validated ground truth labels as a boolean tensor, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor or list[int].
            ValueError: If the labels shape is invalid.

        Examples:
            >>> labels = torch.tensor([0, 1, 1, 0])
            >>> validated_labels = ImageBatchValidator.validate_gt_label(labels)
            >>> validated_labels
            tensor([False,  True,  True, False])
        """
        if labels is None:
            return None
        if isinstance(labels, list):
            labels = torch.tensor(labels)
        if not isinstance(labels, torch.Tensor):
            msg = f"Ground truth labels must be a torch.Tensor or list[int], got {type(labels)}."
            raise TypeError(msg)
        if labels.ndim != 1:
            msg = f"Ground truth labels must be 1-dimensional, got shape {labels.shape}."
            raise ValueError(msg)
        return labels.to(torch.bool)

    @staticmethod
    def validate_gt_mask(masks: torch.Tensor | None) -> Mask | None:
        """Validate a batch of ground truth masks.

        Args:
            masks (torch.Tensor | None): Input batch of ground truth masks.

        Returns:
            Mask | None: Validated batch of ground truth masks, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the mask shape is invalid.

        Examples:
            >>> masks = torch.randint(0, 2, (4, 1, 224, 224))  # Batch of 4 masks
            >>> validated_masks = ImageBatchValidator.validate_gt_mask(masks)
            >>> isinstance(validated_masks, Mask)
            True
            >>> validated_masks.shape
            torch.Size([4, 224, 224])
        """
        if masks is None:
            return None
        if not isinstance(masks, torch.Tensor):
            msg = f"Ground truth masks must be a torch.Tensor, got {type(masks)}."
            raise TypeError(msg)
        if masks.ndim not in {3, 4}:
            msg = f"Ground truth masks must have shape [N, H, W] or [N, 1, H, W], got shape {masks.shape}."
            raise ValueError(msg)
        if masks.ndim == 4 and masks.shape[1] != 1:
            msg = f"Ground truth masks must have 1 channel, got {masks.shape[1]}."
            raise ValueError(msg)

        return Mask(masks.squeeze(1) if masks.ndim == 4 else masks, dtype=torch.bool)

    @staticmethod
    def validate_anomaly_map(anomaly_map: torch.Tensor | np.ndarray | None) -> Mask | None:
        """Validate a batch of anomaly maps.

        Args:
            anomaly_map (torch.Tensor | np.ndarray | None): Input batch of anomaly maps.

        Returns:
            Mask | None: Validated batch of anomaly maps as a Mask, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor or np.ndarray.
            ValueError: If the anomaly map shape is invalid.

        Examples:
            >>> anomaly_map = torch.rand(4, 1, 224, 224)  # Batch of 4 anomaly maps
            >>> validated_map = ImageBatchValidator.validate_anomaly_map(anomaly_map)
            >>> isinstance(validated_map, Mask)
            True
            >>> validated_map.shape
            torch.Size([4, 224, 224])
        """
        if anomaly_map is None:
            return None
        if isinstance(anomaly_map, np.ndarray):
            anomaly_map = torch.from_numpy(anomaly_map)
        if not isinstance(anomaly_map, torch.Tensor):
            msg = f"Anomaly map must be a torch.Tensor or np.ndarray, got {type(anomaly_map)}."
            raise TypeError(msg)
        if anomaly_map.ndim not in {3, 4}:
            msg = f"Anomaly maps must have shape [N, H, W] or [N, 1, H, W], got shape {anomaly_map.shape}."
            raise ValueError(msg)
        if anomaly_map.ndim == 4 and anomaly_map.shape[1] != 1:
            msg = f"Anomaly maps must have 1 channel, got {anomaly_map.shape[1]}."
            raise ValueError(msg)

        return Mask(anomaly_map.squeeze(1) if anomaly_map.ndim == 4 else anomaly_map)

    @staticmethod
    def validate_image_path(image_paths: list[str] | None) -> list[str] | None:
        """Validate the image paths for a batch.

        Args:
            image_paths (list[str] | None): Input image paths.

        Returns:
            list[str] | None: Validated image paths, or None.

        Examples:
            >>> paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
            >>> validated_paths = ImageBatchValidator.validate_image_path(paths)
            >>> validated_paths == paths
            True
        """
        if image_paths is None:
            return None
        return [validate_path(path) for path in image_paths]

    @staticmethod
    def validate_mask_path(mask_path: Sequence[str] | None) -> list[str] | None:
        """Validate the mask paths for a batch.

        Args:
            mask_path (list[str] | None): Input mask paths.

        Returns:
            list[str] | None: Validated mask paths, or None.

        Examples:
            >>> paths = ["/path/to/mask1.png", "/path/to/mask2.png"]
            >>> validated_paths = ImageBatchValidator.validate_mask_path(paths)
            >>> validated_paths == paths
            True
        """
        if mask_path is None:
            return None
        return [validate_path(path) for path in mask_path]

    @staticmethod
    def validate_pred_score(pred_score: torch.Tensor | Sequence[float] | None) -> torch.Tensor | None:
        """Validate the prediction scores for a batch.

        Args:
            pred_score (torch.Tensor | Sequence[float] | None): Input prediction scores.

        Returns:
            torch.Tensor | None: Validated prediction scores as a float32 tensor, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor, Sequence[float], or None.
            ValueError: If the prediction scores are not 1-dimensional.

        Examples:
            >>> scores = torch.tensor([0.8, 0.2, 0.6, 0.4])
            >>> validated_scores = ImageBatchValidator.validate_pred_score(scores)
            >>> validated_scores
            tensor([0.8000, 0.2000, 0.6000, 0.4000])
            >>> scores_list = [0.8, 0.2, 0.6, 0.4]
            >>> validated_scores = ImageBatchValidator.validate_pred_score(scores_list)
            >>> validated_scores
            tensor([0.8000, 0.2000, 0.6000, 0.4000])
            >>> validated_scores = ImageBatchValidator.validate_pred_score(None)
            >>> validated_scores is None
            True
        """
        if pred_score is None:
            return None
        if isinstance(pred_score, Sequence) and not isinstance(pred_score, torch.Tensor):
            pred_score = torch.tensor(pred_score)
        if not isinstance(pred_score, torch.Tensor):
            msg = f"Prediction scores must be a torch.Tensor, Sequence[float], or None, got {type(pred_score)}."
            raise TypeError(msg)
        if pred_score.ndim != 1:
            msg = f"Prediction scores must be 1-dimensional, got shape {pred_score.shape}."
            raise ValueError(msg)
        return pred_score.to(torch.float32)

    @staticmethod
    def validate_pred_label(pred_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction labels for a batch.

        Args:
            pred_label (torch.Tensor | None): Input prediction labels.

        Returns:
            torch.Tensor | None: Validated prediction labels as a boolean tensor, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the prediction labels are not 1-dimensional.

        Examples:
            >>> labels = torch.tensor([1, 0, 1, 0])
            >>> validated_labels = ImageBatchValidator.validate_pred_labels(labels)
            >>> validated_labels
            tensor([ True, False,  True, False])
        """
        if pred_label is None:
            return None
        if not isinstance(pred_label, torch.Tensor):
            msg = f"Predicted labels must be a torch.Tensor, got {type(pred_label)}."
            raise TypeError(msg)
        if pred_label.ndim != 1:
            msg = f"Predicted labels must be 1-dimensional, got shape {pred_label.shape}."
            raise ValueError(msg)
        return pred_label.to(torch.bool)

    @staticmethod
    def validate_pred_mask(pred_mask: torch.Tensor | None) -> Mask | None:
        """Validate a batch of prediction masks.

        Args:
            pred_mask (torch.Tensor | None): Input batch of prediction masks.

        Returns:
            Mask | None: Validated batch of prediction masks, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the mask shape is invalid.

        Examples:
            >>> pred_mask = torch.randint(0, 2, (4, 1, 224, 224))  # Batch of 4 prediction masks
            >>> validated_mask = ImageBatchValidator.validate_pred_mask(pred_mask)
            >>> isinstance(validated_mask, Mask)
            True
            >>> validated_mask.shape
            torch.Size([4, 224, 224])
        """
        return ImageBatchValidator.validate_gt_mask(pred_mask)  # We can reuse the gt_mask validation
