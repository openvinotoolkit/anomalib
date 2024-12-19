"""Validate PyTorch tensor data for depth maps.

This module provides validators for depth data stored as PyTorch tensors. The validators
ensure data consistency and correctness for depth maps and their batches.

The validators check:
    - Tensor shapes and dimensions
    - Data types
    - Value ranges
    - Label formats
    - Mask properties
    - Path validity

Example:
    Validate a single depth map::

        >>> from anomalib.data.validators import DepthValidator
        >>> validator = DepthValidator()
        >>> validator.validate_depth_map(depth_map)

    Validate a batch of depth maps::

        >>> from anomalib.data.validators import DepthBatchValidator
        >>> validator = DepthBatchValidator()
        >>> validator(depth_maps=depth_maps, labels=labels, masks=masks)

Note:
    The validators are used internally by the data modules to ensure data
    consistency before processing depth data.
"""

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
    """Validate torch.Tensor data for depth images.

    This class provides validation methods for depth data stored as PyTorch tensors.
    It ensures data consistency and correctness for depth maps and associated metadata.

    The validator checks:
        - Tensor shapes and dimensions
        - Data types
        - Value ranges
        - Label formats
        - Mask properties
        - Path validity

    Example:
        Validate a depth map and associated metadata::

            >>> from anomalib.data.validators import DepthValidator
            >>> validator = DepthValidator()
            >>> depth_map = torch.rand(224, 224)  # [H, W]
            >>> validated_map = validator.validate_depth_map(depth_map)
            >>> label = 1
            >>> validated_label = validator.validate_gt_label(label)
            >>> mask = torch.randint(0, 2, (1, 224, 224))  # [1, H, W]
            >>> validated_mask = validator.validate_gt_mask(mask)

    Note:
        The validator is used internally by the data modules to ensure data
        consistency before processing.
    """

    @staticmethod
    def validate_image(image: torch.Tensor) -> Image:
        """Validate the image tensor.

        This method validates and normalizes input image tensors. It handles:
            - RGB images only
            - Channel-first format [C, H, W]
            - Type conversion to float32
            - Value range normalization

        Args:
            image (``torch.Tensor``): Input image tensor to validate.

        Returns:
            ``Image``: Validated image as a torchvision Image object.

        Raises:
            TypeError: If ``image`` is not a torch.Tensor.
            ValueError: If ``image`` dimensions or channels are invalid.

        Example:
            Validate RGB image::

                >>> import torch
                >>> from anomalib.data.validators import DepthValidator
                >>> image = torch.rand(3, 256, 256)  # [C, H, W]
                >>> validated = DepthValidator.validate_image(image)
                >>> validated.shape
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

        This method validates and normalizes input labels. It handles:
            - Integer and tensor inputs
            - Type conversion to boolean
            - Scalar values only

        Args:
            label (``int`` | ``torch.Tensor`` | ``None``): Input ground truth label.

        Returns:
            ``torch.Tensor`` | ``None``: Validated ground truth label as boolean tensor.

        Raises:
            TypeError: If ``label`` is neither an integer nor a torch.Tensor.
            ValueError: If ``label`` shape is invalid.

        Example:
            Validate integer and tensor labels::

                >>> from anomalib.data.validators import DepthValidator
                >>> label_int = 1
                >>> validated = DepthValidator.validate_gt_label(label_int)
                >>> validated
                tensor(True)
                >>> label_tensor = torch.tensor(0)
                >>> validated = DepthValidator.validate_gt_label(label_tensor)
                >>> validated
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

        This method validates and normalizes input masks. It handles:
            - 2D and 3D inputs
            - Single-channel masks
            - Type conversion to boolean
            - Channel dimension squeezing

        Args:
            mask (``torch.Tensor`` | ``None``): Input ground truth mask.

        Returns:
            ``Mask`` | ``None``: Validated ground truth mask as torchvision Mask.

        Raises:
            TypeError: If ``mask`` is not a torch.Tensor.
            ValueError: If ``mask`` dimensions or channels are invalid.

        Example:
            Validate binary segmentation mask::

                >>> import torch
                >>> from anomalib.data.validators import DepthValidator
                >>> mask = torch.randint(0, 2, (1, 224, 224))  # [1, H, W]
                >>> validated = DepthValidator.validate_gt_mask(mask)
                >>> isinstance(validated, Mask)
                True
                >>> validated.shape
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

        This method validates input image file paths.

        Args:
            image_path (``str`` | ``None``): Input image path to validate.

        Returns:
            ``str`` | ``None``: Validated image path, or None.

        Example:
            Validate image file path::

                >>> from anomalib.data.validators import DepthValidator
                >>> path = "/path/to/image.jpg"
                >>> validated = DepthValidator.validate_image_path(path)
                >>> validated == path
                True
        """
        return validate_path(image_path) if image_path else None

    @staticmethod
    def validate_depth_map(depth_map: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the depth map.

        This method validates and normalizes input depth maps. It handles:
            - 2D and 3D inputs
            - Single and multi-channel depth maps
            - Type conversion to float32

        Args:
            depth_map (``torch.Tensor`` | ``None``): Input depth map to validate.

        Returns:
            ``torch.Tensor`` | ``None``: Validated depth map as float32 tensor.

        Raises:
            TypeError: If ``depth_map`` is not a torch.Tensor.
            ValueError: If ``depth_map`` dimensions or channels are invalid.

        Example:
            Validate single-channel depth map::

                >>> import torch
                >>> from anomalib.data.validators import DepthValidator
                >>> depth_map = torch.rand(224, 224)  # [H, W]
                >>> validated = DepthValidator.validate_depth_map(depth_map)
                >>> validated.shape
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

        This method validates input depth map file paths.

        Args:
            depth_path (``str`` | ``None``): Input depth path to validate.

        Returns:
            ``str`` | ``None``: Validated depth path, or None.

        Example:
            Validate depth map file path::

                >>> from anomalib.data.validators import DepthValidator
                >>> path = "/path/to/depth.png"
                >>> validated = DepthValidator.validate_depth_path(path)
                >>> validated == path
                True
        """
        return validate_path(depth_path) if depth_path else None

    @staticmethod
    def validate_anomaly_map(anomaly_map: torch.Tensor | None) -> Mask | None:
        """Validate the anomaly map."""
        return ImageValidator.validate_anomaly_map(anomaly_map)

    @staticmethod
    def validate_pred_score(pred_score: torch.Tensor | float | None) -> torch.Tensor | None:
        """Validate the prediction score."""
        return ImageValidator.validate_pred_score(pred_score)

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

    @staticmethod
    def validate_explanation(explanation: str | None) -> str | None:
        """Validate the explanation."""
        return ImageValidator.validate_explanation(explanation)


class DepthBatchValidator:
    """Validate torch.Tensor data for batches of depth images.

    This class provides validation methods for batches of depth data stored as PyTorch tensors.
    It ensures data consistency and correctness for depth maps and associated metadata.

    The validator checks:
        - Tensor shapes and dimensions
        - Data types
        - Value ranges
        - Label formats
        - Mask properties
        - Path validity

    Example:
        Validate a batch of depth maps and associated metadata::

            >>> from anomalib.data.validators import DepthBatchValidator
            >>> validator = DepthBatchValidator()
            >>> depth_maps = torch.rand(32, 224, 224)  # [N, H, W]
            >>> labels = torch.zeros(32)
            >>> masks = torch.zeros((32, 224, 224))
            >>> validated_maps = validator.validate_depth_map(depth_maps)
            >>> validated_labels = validator.validate_gt_label(labels)
            >>> validated_masks = validator.validate_gt_mask(masks)

    Note:
        The validator is used internally by the data modules to ensure data
        consistency before processing.
    """

    @staticmethod
    def validate_image(image: torch.Tensor) -> Image:
        """Validate the image tensor for a batch.

        This method validates batches of images stored as PyTorch tensors. It handles:
            - Channel-first format [N, C, H, W]
            - RGB images only
            - Type conversion to float32
            - Value range normalization

        Args:
            image (``torch.Tensor``): Input image tensor to validate.

        Returns:
            ``Image``: Validated image as a torchvision Image object.

        Raises:
            TypeError: If ``image`` is not a torch.Tensor.
            ValueError: If ``image`` dimensions or channels are invalid.

        Example:
            >>> import torch
            >>> from anomalib.data.validators import DepthBatchValidator
            >>> image = torch.rand(32, 3, 256, 256)  # [N, C, H, W]
            >>> validated = DepthBatchValidator.validate_image(image)
            >>> validated.shape
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

        This method validates ground truth labels for batches. It handles:
            - Conversion to boolean tensor
            - Batch dimension validation
            - None inputs

        Args:
            gt_label (``torch.Tensor`` | ``Sequence[int]`` | ``None``): Input ground truth
                label to validate.

        Returns:
            ``torch.Tensor`` | ``None``: Validated ground truth label as a boolean tensor,
                or None.

        Raises:
            TypeError: If ``gt_label`` is not a sequence of integers or torch.Tensor.
            ValueError: If ``gt_label`` does not match expected batch size or data type.

        Example:
            >>> import torch
            >>> from anomalib.data.validators import DepthBatchValidator
            >>> gt_label = torch.tensor([0, 1, 1, 0])
            >>> validated = DepthBatchValidator.validate_gt_label(gt_label)
            >>> print(validated)
            tensor([False,  True,  True, False])
        """
        return ImageBatchValidator.validate_gt_label(gt_label)

    @staticmethod
    def validate_gt_mask(gt_mask: torch.Tensor | None) -> Mask | None:
        """Validate the ground truth mask for a batch.

        This method validates ground truth masks for batches. It handles:
            - Batch dimension validation
            - Binary mask values
            - None inputs

        Args:
            gt_mask (``torch.Tensor`` | ``None``): Input ground truth mask to validate.

        Returns:
            ``Mask`` | ``None``: Validated ground truth mask as a torchvision Mask object,
                or None.

        Raises:
            TypeError: If ``gt_mask`` is not a torch.Tensor.
            ValueError: If ``gt_mask`` shape or batch size is invalid.

        Example:
            >>> import torch
            >>> from anomalib.data.validators import DepthBatchValidator
            >>> gt_mask = torch.randint(0, 2, (4, 224, 224))  # [N, H, W]
            >>> validated = DepthBatchValidator.validate_gt_mask(gt_mask)
            >>> print(validated.shape)
            torch.Size([4, 224, 224])
        """
        return ImageBatchValidator.validate_gt_mask(gt_mask)

    @staticmethod
    def validate_mask_path(mask_path: Sequence[str] | None) -> list[str] | None:
        """Validate the mask paths for a batch.

        This method validates file paths for batches of mask images. It handles:
            - Path existence validation
            - Batch size consistency
            - None inputs

        Args:
            mask_path (``Sequence[str]`` | ``None``): Input sequence of mask paths.

        Returns:
            ``list[str]`` | ``None``: Validated list of mask paths, or None.

        Raises:
            TypeError: If ``mask_path`` is not a sequence of strings.
            ValueError: If number of paths does not match expected batch size.

        Example:
            >>> from anomalib.data.validators import DepthBatchValidator
            >>> paths = ["path/to/mask_1.png", "path/to/mask_2.png"]
            >>> validated = DepthBatchValidator.validate_mask_path(paths)
            >>> print(validated)
            ['path/to/mask_1.png', 'path/to/mask_2.png']
        """
        return ImageBatchValidator.validate_mask_path(mask_path)

    @staticmethod
    def validate_image_path(image_path: list[str] | None) -> list[str] | None:
        """Validate the image paths for a batch.

        This method validates file paths for batches of images. It handles:
            - Path existence validation
            - Batch size consistency
            - None inputs

        Args:
            image_path (``list[str]`` | ``None``): Input list of image paths.

        Returns:
            ``list[str]`` | ``None``: Validated list of image paths, or None.

        Raises:
            TypeError: If ``image_path`` is not a list of strings.

        Example:
            >>> from anomalib.data.validators import DepthBatchValidator
            >>> paths = ["path/to/image_1.jpg", "path/to/image_2.jpg"]
            >>> validated = DepthBatchValidator.validate_image_path(paths)
            >>> print(validated)
            ['path/to/image_1.jpg', 'path/to/image_2.jpg']
        """
        return ImageBatchValidator.validate_image_path(image_path)

    @staticmethod
    def validate_depth_map(depth_map: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the depth map for a batch.

        This method validates batches of depth maps. It handles:
            - Single-channel and RGB depth maps
            - Batch dimension validation
            - Type conversion to float32
            - None inputs

        Args:
            depth_map (``torch.Tensor`` | ``None``): Input depth map to validate.

        Returns:
            ``torch.Tensor`` | ``None``: Validated depth map as float32, or None.

        Raises:
            TypeError: If ``depth_map`` is not a torch.Tensor.
            ValueError: If ``depth_map`` shape is invalid or batch size mismatch.

        Example:
            >>> import torch
            >>> from anomalib.data.validators import DepthBatchValidator
            >>> depth_map = torch.rand(4, 224, 224)  # [N, H, W]
            >>> validated = DepthBatchValidator.validate_depth_map(depth_map)
            >>> print(validated.shape)
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

        This method validates file paths for batches of depth maps. It handles:
            - Path existence validation
            - Batch size consistency
            - None inputs

        Args:
            depth_path (``list[str]`` | ``None``): Input list of depth paths.

        Returns:
            ``list[str]`` | ``None``: Validated list of depth paths, or None.

        Raises:
            TypeError: If ``depth_path`` is not a list of strings.

        Example:
            >>> from anomalib.data.validators import DepthBatchValidator
            >>> paths = ["path/to/depth_1.png", "path/to/depth_2.png"]
            >>> validated = DepthBatchValidator.validate_depth_path(paths)
            >>> print(validated)
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
        pred_score: torch.Tensor | np.ndarray | float | None,
    ) -> torch.Tensor | None:
        """Validate the prediction scores for a batch."""
        return ImageBatchValidator.validate_pred_score(pred_score)

    @staticmethod
    def validate_pred_mask(pred_mask: torch.Tensor | None) -> Mask | None:
        """Validate the prediction mask for a batch."""
        return ImageBatchValidator.validate_pred_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction label for a batch."""
        return ImageBatchValidator.validate_pred_label(pred_label)

    @staticmethod
    def validate_explanation(explanation: list[str] | None) -> list[str] | None:
        """Validate the explanations for a batch."""
        return ImageBatchValidator.validate_explanation(explanation)
