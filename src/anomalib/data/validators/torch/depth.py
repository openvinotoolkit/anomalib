"""Validate torch depth data."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np
from torchvision.transforms.v2.functional import to_dtype_image
from torchvision.tv_tensors import Image, Mask

import torch
from anomalib.data.validators.path import validate_path
from anomalib.data.validators.torch.image import ImageBatchValidator, ImageValidator


class DepthValidator:
    """Validate torch.Tensor data for depth images."""

    @staticmethod
    def validate_image(image: torch.Tensor) -> Image:
        """Validate the image tensor.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            Image: Validated image as a torchvision Image object.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the image tensor does not have the correct shape.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import DepthValidator
            >>> image = torch.rand(3, 256, 256)
            >>> validated_image = DepthValidator.validate_image(image)
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
        return Image(to_dtype_image(image, torch.float32, scale=True))

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
            >>> from anomalib.data.validators import DepthValidator
            >>> label_int = 1
            >>> validated_label = DepthValidator.validate_gt_label(label_int)
            >>> validated_label
            tensor(True)
            >>> label_tensor = torch.tensor(0)
            >>> validated_label = DepthValidator.validate_gt_label(label_tensor)
            >>> validated_label
            tensor(False)
        """
        if label is None:
            return None
        if isinstance(label, int | np.integer):
            label = torch.tensor(int(label))
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
            >>> from anomalib.data.validators import DepthValidator
            >>> mask = torch.randint(0, 2, (1, 224, 224))
            >>> validated_mask = DepthValidator.validate_gt_mask(mask)
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
    def validate_image_path(image_path: str | None) -> str | None:
        """Validate the image path.

        Args:
            image_path (str | None): Input image path.

        Returns:
            str | None: Validated image path, or None.

        Examples:
            >>> from anomalib.data.validators import DepthValidator
            >>> path = "/path/to/image.jpg"
            >>> validated_path = DepthValidator.validate_image_path(path)
            >>> validated_path == path
            True
        """
        return validate_path(image_path) if image_path else None

    @staticmethod
    def validate_depth_map(depth_map: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the depth map.

        Args:
            depth_map (torch.Tensor | None): Input depth map.

        Returns:
            torch.Tensor | None: Validated depth map, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the depth map shape is invalid.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import DepthValidator
            >>> depth_map = torch.rand(224, 224)
            >>> validated_map = DepthValidator.validate_depth_map(depth_map)
            >>> validated_map.shape
            torch.Size([224, 224])
        """
        if depth_map is None:
            return None
        if not isinstance(depth_map, torch.Tensor):
            msg = f"Depth map must be a torch.Tensor, got {type(depth_map)}."
            raise TypeError(msg)
        if depth_map.ndim not in {2, 3}:
            msg = f"Depth map must have shape [H, W] or [C, H, W], got shape {depth_map.shape}."
            raise ValueError(msg)
        if depth_map.ndim == 3 and depth_map.shape[0] not in {1, 3}:
            msg = f"Depth map with 3 dimensions must have 1 or 3 channels, got {depth_map.shape[0]}."
            raise ValueError(msg)
        return depth_map.to(torch.float32)

    @staticmethod
    def validate_depth_path(depth_path: str | None) -> str | None:
        """Validate the depth path.

        Args:
            depth_path (str | None): Input depth path.

        Returns:
            str | None: Validated depth path, or None.

        Examples:
            >>> from anomalib.data.validators import DepthValidator
            >>> path = "/path/to/depth.png"
            >>> validated_path = DepthValidator.validate_depth_path(path)
            >>> validated_path == path
            True
        """
        return validate_path(depth_path) if depth_path else None

    @staticmethod
    def validate_anomaly_map(anomaly_map: torch.Tensor | None) -> Mask | None:
        """Validate the anomaly map."""
        return ImageValidator.validate_anomaly_map(anomaly_map)

    @staticmethod
    def validate_pred_score(
        pred_score: torch.Tensor | float | None,
        anomaly_map: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Validate the prediction score."""
        return ImageValidator.validate_pred_score(pred_score, anomaly_map)

    @staticmethod
    def validate_pred_mask(pred_mask: torch.Tensor | None) -> Mask | None:
        """Validate the prediction mask."""
        return ImageValidator.validate_pred_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction label."""
        return ImageValidator.validate_pred_label(pred_label)

    @staticmethod
    def validate_mask_path(mask_path: str | None) -> str | None:
        """Validate the mask path."""
        return ImageValidator.validate_mask_path(mask_path)


class DepthBatchValidator:
    """Validate torch.Tensor data for batches of depth images."""

    @staticmethod
    def validate_image(image: torch.Tensor) -> Image:
        """Validate the image tensor for a batch.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            Image: Validated image as a torchvision Image object.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the image tensor does not have the correct shape.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import DepthBatchValidator
            >>> image = torch.rand(32, 3, 256, 256)
            >>> validated_image = DepthBatchValidator.validate_image(image)
            >>> validated_image.shape
            torch.Size([32, 3, 256, 256])
        """
        if not isinstance(image, torch.Tensor):
            msg = f"Image must be a torch.Tensor, got {type(image)}."
            raise TypeError(msg)
        if image.ndim != 4:
            msg = f"Image must have shape [N, C, H, W], got shape {image.shape}."
            raise ValueError(msg)
        if image.shape[1] != 3:
            msg = f"Image must have 3 channels, got {image.shape[1]}."
            raise ValueError(msg)
        return Image(to_dtype_image(image, torch.float32, scale=True))

    @staticmethod
    def validate_gt_label(gt_label: torch.Tensor | Sequence[int] | None) -> torch.Tensor | None:
        """Validate the ground truth label for a batch.

        Args:
            gt_label (torch.Tensor | Sequence[int] | None): Input ground truth label.

        Returns:
            torch.Tensor | None: Validated ground truth label as a boolean tensor, or None.

        Raises:
            TypeError: If the input is not a sequence of integers or a torch.Tensor.
            ValueError: If the ground truth label does not match the expected batch size or data type.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import DepthBatchValidator
            >>> gt_label = torch.tensor([0, 1, 1, 0])
            >>> validated_label = DepthBatchValidator.validate_gt_label(gt_label)
            >>> print(validated_label)
            tensor([False,  True,  True, False])
        """
        return ImageBatchValidator.validate_gt_label(gt_label)

    @staticmethod
    def validate_gt_mask(gt_mask: torch.Tensor | None) -> Mask | None:
        """Validate the ground truth mask for a batch.

        Args:
            gt_mask (torch.Tensor | None): Input ground truth mask.

        Returns:
            Mask | None: Validated ground truth mask as a torchvision Mask object, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the ground truth mask does not have the correct shape or batch size.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import DepthBatchValidator
            >>> gt_mask = torch.randint(0, 2, (4, 224, 224))
            >>> validated_mask = DepthBatchValidator.validate_gt_mask(gt_mask)
            >>> print(validated_mask.shape)
            torch.Size([4, 224, 224])
        """
        return ImageBatchValidator.validate_gt_mask(gt_mask)

    @staticmethod
    def validate_mask_path(mask_path: Sequence[str] | None) -> list[str] | None:
        """Validate the mask paths for a batch.

        Args:
            mask_path (Sequence[str] | None): Input sequence of mask paths.

        Returns:
            list[str] | None: Validated list of mask paths, or None.

        Raises:
            TypeError: If the input is not a sequence of strings.
            ValueError: If the number of mask paths does not match the expected batch size.

        Examples:
            >>> from anomalib.data.validators import DepthBatchValidator
            >>> mask_paths = ["path/to/mask_1.png", "path/to/mask_2.png"]
            >>> validated_paths = DepthBatchValidator.validate_mask_path(mask_paths)
            >>> print(validated_paths)
            ['path/to/mask_1.png', 'path/to/mask_2.png']
        """
        return ImageBatchValidator.validate_mask_path(mask_path)

    @staticmethod
    def validate_image_path(image_path: list[str] | None) -> list[str] | None:
        """Validate the image paths for a batch.

        Args:
            image_path (list[str] | None): Input list of image paths.

        Returns:
            list[str] | None: Validated list of image paths, or None.

        Raises:
            TypeError: If the input is not a list of strings.

        Examples:
            >>> from anomalib.data.validators import DepthBatchValidator
            >>> image_paths = ["path/to/image_1.jpg", "path/to/image_2.jpg"]
            >>> validated_paths = DepthBatchValidator.validate_image_path(image_paths)
            >>> print(validated_paths)
            ['path/to/image_1.jpg', 'path/to/image_2.jpg']
        """
        return ImageBatchValidator.validate_image_path(image_path)

    @staticmethod
    def validate_depth_map(depth_map: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the depth map for a batch.

        Args:
            depth_map (torch.Tensor | None): Input depth map.

        Returns:
            torch.Tensor | None: Validated depth map, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the depth map shape is invalid or doesn't match the batch size.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import DepthBatchValidator
            >>> depth_map = torch.rand(4, 224, 224)
            >>> validated_map = DepthBatchValidator.validate_depth_map(depth_map)
            >>> print(validated_map.shape)
            torch.Size([4, 224, 224])
        """
        if depth_map is None:
            return None
        if not isinstance(depth_map, torch.Tensor):
            msg = f"Depth map must be a torch.Tensor, got {type(depth_map)}."
            raise TypeError(msg)
        if depth_map.ndim not in {3, 4}:
            msg = f"Depth map must have shape [N, H, W] or [N, C, H, W], got shape {depth_map.shape}."
            raise ValueError(msg)
        if depth_map.ndim == 4 and depth_map.shape[1] != 1 and depth_map.shape[1] != 3:
            msg = f"Depth map with 4 dimensions must have 1 or 3 channels, got {depth_map.shape[1]}."
            raise ValueError(msg)
        return depth_map.to(torch.float32)

    @staticmethod
    def validate_depth_path(depth_path: list[str] | None) -> list[str] | None:
        """Validate the depth paths for a batch.

        Args:
            depth_path (list[str] | None): Input list of depth paths.

        Returns:
            list[str] | None: Validated list of depth paths, or None.

        Raises:
            TypeError: If the input is not a list of strings.

        Examples:
            >>> from anomalib.data.validators import DepthBatchValidator
            >>> depth_paths = ["path/to/depth_1.png", "path/to/depth_2.png"]
            >>> validated_paths = DepthBatchValidator.validate_depth_path(depth_paths)
            >>> print(validated_paths)
            ['path/to/depth_1.png', 'path/to/depth_2.png']
        """
        if depth_path is None:
            return None
        if not isinstance(depth_path, list):
            msg = f"Depth path must be a list of strings, got {type(depth_path)}."
            raise TypeError(msg)
        return [validate_path(path) for path in depth_path]

    @staticmethod
    def validate_anomaly_map(anomaly_map: torch.Tensor | np.ndarray | None) -> Mask | None:
        """Validate the anomaly map for a batch."""
        return ImageBatchValidator.validate_anomaly_map(anomaly_map)

    @staticmethod
    def validate_pred_score(
        pred_score: torch.Tensor | Sequence[float] | None,
        anomaly_map: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Validate the prediction scores for a batch."""
        return ImageBatchValidator.validate_pred_score(pred_score, anomaly_map)

    @staticmethod
    def validate_pred_mask(pred_mask: torch.Tensor | None) -> Mask | None:
        """Validate the prediction mask for a batch."""
        return ImageBatchValidator.validate_pred_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction label for a batch."""
        return ImageBatchValidator.validate_pred_label(pred_label)
