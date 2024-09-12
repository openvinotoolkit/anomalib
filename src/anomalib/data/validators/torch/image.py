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
    def validate_image(image: torch.Tensor) -> torch.Tensor:
        """Validate the image tensor.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Validated image tensor.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the image tensor does not have the correct shape.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import ImageValidator
            >>> image = torch.rand(3, 256, 256)
            >>> validated_image = ImageValidator.validate_image(image)
            >>> validated_image.shape
            torch.Size([3, 256, 256])
        """
        if not isinstance(image, torch.Tensor):
            msg = f"Image must be a torch.Tensor, got {type(image)}."
            raise TypeError(msg)
        if image.ndim != 3:
            msg = f"Image must have shape [C, H, W], got shape {image.shape}."
            raise ValueError(msg)
        if image.shape[0] != 3:
            msg = f"Image must have 3 channels, got {image.shape[0]}."
            raise ValueError(msg)
        return to_dtype_image(image, torch.float32, scale=True)

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
            label = torch.tensor(label)
        if not isinstance(label, torch.Tensor):
            msg = f"Ground truth label must be an integer or a torch.Tensor, got {type(label)}."
            raise TypeError(msg)
        if label.ndim != 0:
            msg = f"Ground truth label must be a scalar, got shape {label.shape}."
            raise ValueError(msg)
        if torch.is_floating_point(label):
            msg = f"Ground truth label must be boolean or integer, got {label.dtype}."
            raise TypeError(msg)
        return label.bool()

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
            msg = f"Mask must be a torch.Tensor, got {type(mask)}."
            raise TypeError(msg)
        if mask.ndim not in {2, 3}:
            msg = f"Mask must have shape [H, W] or [1, H, W] got shape {mask.shape}."
            raise ValueError(msg)
        if mask.ndim == 3:
            if mask.shape[0] != 1:
                msg = f"Mask must have 1 channel, got {mask.shape[0]}."
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
                msg = f"Anomaly map with 3 dimensions must have 1 channel, got {anomaly_map.shape[0]}."
                raise ValueError(msg)
            anomaly_map = anomaly_map.squeeze(0)

        return Mask(anomaly_map, dtype=torch.float32)

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
    def validate_pred_score(
        pred_score: torch.Tensor | float | None,
        anomaly_map: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Validate the prediction score.

        Args:
            pred_score (torch.Tensor | float | None): Input prediction score.
            anomaly_map (torch.Tensor | None): Input anomaly map.

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
            return torch.amax(anomaly_map, dim=(-2, -1)) if anomaly_map else None

        if not isinstance(pred_score, torch.Tensor):
            try:
                pred_score = torch.tensor(pred_score)
            except Exception as e:
                msg = "Failed to convert pred_score to a torch.Tensor."
                raise ValueError(msg) from e
        pred_score = pred_score.squeeze()
        if pred_score.ndim != 0:
            msg = f"Predicted score must be a scalar, got shape {pred_score.shape}."
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
            try:
                pred_label = torch.tensor(pred_label)
            except Exception as e:
                msg = "Failed to convert pred_score to a torch.Tensor."
                raise ValueError(msg) from e
        pred_label = pred_label.squeeze()
        if pred_label.ndim != 0:
            msg = f"Predicted label must be a scalar, got shape {pred_label.shape}."
            raise ValueError(msg)
        return pred_label.to(torch.bool)


class ImageBatchValidator:
    """Validate torch.Tensor data for batches of images."""

    @staticmethod
    def validate_image(image: torch.Tensor) -> Image:
        """Validate the image for a batch."""
        assert isinstance(image, torch.Tensor), f"Image must be a torch.Tensor, got {type(image)}."
        assert image.ndim in {3, 4}, f"Image must have shape [C, H, W] or [N, C, H, W], got shape {image.shape}."
        if image.ndim == 3:
            image = image.unsqueeze(0)  # add batch dimension
        assert image.shape[1] == 3, f"Image must have 3 channels, got {image.shape[0]}."
        return Image(image, dtype=torch.float32)

    @staticmethod
    def validate_gt_label(gt_label: torch.Tensor | Sequence[int] | None, batch_size: int) -> torch.Tensor | None:
        """Validate the ground truth label for a batch."""
        if gt_label is None:
            return None
        if isinstance(gt_label, Sequence):
            gt_label = torch.tensor(gt_label)
        assert isinstance(
            gt_label,
            torch.Tensor,
        ), f"Ground truth label must be a sequence of integers or a torch.Tensor, got {type(gt_label)}."
        assert gt_label.ndim == 1, f"Ground truth label must be a 1-dimensional vector, got shape {gt_label.shape}."
        assert (
            len(gt_label) == batch_size
        ), f"Ground truth label must have length {batch_size}, got length {len(gt_label)}."
        assert not torch.is_floating_point(gt_label), f"Ground truth label must be boolean or integer, got {gt_label}."
        return gt_label.bool()

    @staticmethod
    def validate_gt_mask(gt_mask: torch.Tensor | None, batch_size: int) -> Mask | None:
        """Validate the ground truth mask for a batch."""
        if gt_mask is None:
            return None
        assert isinstance(gt_mask, torch.Tensor), f"Ground truth mask must be a torch.Tensor, got {type(gt_mask)}."
        assert gt_mask.ndim in {
            2,
            3,
            4,
        }, f"Ground truth mask must have shape [H, W] or [N, H, W] or [N, 1, H, W] got shape {gt_mask.shape}."
        if gt_mask.ndim == 2:
            assert (
                batch_size == 1
            ), f"Invalid shape for gt_mask. Got mask shape {gt_mask.shape} for batch size {batch_size}."
            gt_mask = gt_mask.unsqueeze(0)
        if gt_mask.ndim == 3:
            assert (
                gt_mask.shape[0] == batch_size
            ), f"Invalid shape for gt_mask. Got mask shape {gt_mask.shape} for batch size {batch_size}."
        if gt_mask.ndim == 4:
            assert gt_mask.shape[1] == 1, f"Ground truth mask must have 1 channel, got {gt_mask.shape[1]}."
            gt_mask = gt_mask.squeeze(1)
        return Mask(gt_mask, dtype=torch.bool)

    @staticmethod
    def validate_mask_path(mask_path: Sequence[str] | None, batch_size: int) -> list[str] | None:
        """Validate the mask paths for a batch."""
        if mask_path is None:
            return None
        assert isinstance(
            mask_path,
            Sequence,
        ), f"Mask path must be a sequence of paths or strings, got {type(mask_path)}."
        assert (
            len(mask_path) == batch_size
        ), f"Invalid length for mask_path. Got length {len(mask_path)} for batch size {batch_size}."
        return [str(path) for path in mask_path]

    @staticmethod
    def validate_anomaly_map(anomaly_map: torch.Tensor | np.ndarray | None, batch_size: int) -> Mask | None:
        """Validate the anomaly map for a batch."""
        if anomaly_map is None:
            return None
        if not isinstance(anomaly_map, torch.Tensor):
            try:
                anomaly_map = torch.tensor(anomaly_map)
            except Exception as e:
                msg = "Failed to convert anomaly_map to a torch.Tensor."
                raise ValueError(msg) from e
        assert anomaly_map.ndim in {
            2,
            3,
            4,
        }, f"Anomaly map must have shape [H, W] or [N, H, W] or [N, 1, H, W], got shape {anomaly_map.shape}."
        if anomaly_map.ndim == 2:
            assert (
                batch_size == 1
            ), f"Invalid shape for anomaly_map. Got mask shape {anomaly_map.shape} for batch size {batch_size}."
            anomaly_map = anomaly_map.unsqueeze(0)
        if anomaly_map.ndim == 4:
            assert anomaly_map.shape[1] == 1, f"Anomaly map must have 1 channel, got {anomaly_map.shape[1]}."
            anomaly_map = anomaly_map.squeeze(1)
        return Mask(anomaly_map, dtype=torch.float32)

    @staticmethod
    def validate_pred_score(
        pred_score: torch.Tensor | Sequence[float] | None,
        anomaly_map: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Validate the prediction scores for a batch."""
        if pred_score is None:
            return torch.amax(anomaly_map, dim=(-2, -1)) if anomaly_map is not None else None

        if isinstance(pred_score, Sequence) and not isinstance(pred_score, torch.Tensor):
            pred_score = torch.tensor(pred_score)
        assert isinstance(
            pred_score,
            torch.Tensor,
        ), f"Prediction scores must be a torch.Tensor or Sequence[float], got {type(pred_score)}."
        assert pred_score.ndim == 1, f"Prediction scores must be 1-dimensional, got shape {pred_score.shape}."
        return pred_score.to(torch.float32)

    @staticmethod
    def validate_pred_mask(pred_mask: torch.Tensor | None, batch_size: int) -> Mask | None:
        """Validate the prediction mask for a batch."""
        return ImageBatchValidator.validate_gt_mask(pred_mask, batch_size)  # We can reuse the gt_mask validation

    @staticmethod
    def validate_pred_label(pred_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction label for a batch."""
        if pred_label is None:
            return None
        assert isinstance(pred_label, torch.Tensor), f"Predicted label must be a torch.Tensor, got {type(pred_label)}."
        assert pred_label.ndim == 1, f"Predicted label must be 1-dimensional, got shape {pred_label.shape}."
        return pred_label.to(torch.bool)

    @staticmethod
    def validate_image_path(image_path: list[str] | None) -> list[str] | None:
        """Validate the image paths for a batch."""
        if image_path is None:
            return None
        assert isinstance(image_path, list), f"Image path must be a list of strings, got {type(image_path)}."
        return [str(path) for path in image_path]
