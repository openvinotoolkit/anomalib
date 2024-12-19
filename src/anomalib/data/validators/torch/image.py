"""Validate PyTorch tensor data for images.

This module provides validators for image data stored as PyTorch tensors. The validators
ensure data consistency and correctness for images and their batches.

The validators check:
    - Tensor shapes and dimensions
    - Data types
    - Value ranges
    - Label formats
    - Mask properties
    - Path validity

Example:
    Validate a single image::

        >>> from anomalib.data.validators import ImageValidator
        >>> validator = ImageValidator()
        >>> validator.validate_image(image)

    Validate a batch of images::

        >>> from anomalib.data.validators import ImageBatchValidator
        >>> validator = ImageBatchValidator()
        >>> validator(images=images, labels=labels, masks=masks)

Note:
    The validators are used internally by the data modules to ensure data
    consistency before processing.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np
from torchvision.transforms.v2.functional import to_dtype_image
from torchvision.tv_tensors import Image, Mask

import torch
from anomalib.data.validators.path import validate_path


class ImageValidator:
    """Validate torch.Tensor data for images.

    This class provides validation methods for image data stored as PyTorch tensors.
    It ensures data consistency and correctness for images and associated metadata.

    The validator checks:
        - Tensor shapes and dimensions
        - Data types
        - Value ranges
        - Label formats
        - Mask properties
        - Path validity

    Example:
        Validate an image and associated metadata::

            >>> from anomalib.data.validators import ImageValidator
            >>> validator = ImageValidator()
            >>> image = torch.rand(3, 224, 224)  # [C, H, W]
            >>> validated_image = validator.validate_image(image)
            >>> label = 1
            >>> validated_label = validator.validate_gt_label(label)
            >>> mask = torch.randint(0, 2, (1, 224, 224))  # [1, H, W]
            >>> validated_mask = validator.validate_gt_mask(mask)

    Note:
        The validator is used internally by the data modules to ensure data
        consistency before processing.
    """

    @staticmethod
    def validate_image(image: torch.Tensor) -> torch.Tensor:
        """Validate the image tensor.

        This method validates and normalizes input image tensors. It handles:
            - RGB images only
            - Channel-first format [C, H, W]
            - Type conversion to float32
            - Value range normalization

        Args:
            image (``torch.Tensor``): Input image tensor to validate.

        Returns:
            ``torch.Tensor``: Validated image tensor in [C, H, W] format.

        Raises:
            TypeError: If ``image`` is not a torch.Tensor.
            ValueError: If ``image`` dimensions or channels are invalid.

        Example:
            >>> import torch
            >>> from anomalib.data.validators import ImageValidator
            >>> image = torch.rand(3, 256, 256)  # [C, H, W]
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

        This method validates and normalizes input labels. It handles:
            - Integer and tensor inputs
            - Type conversion to boolean
            - Output is always boolean type
            - None inputs return None

        Args:
            label (``int`` | ``torch.Tensor`` | ``None``): Input ground truth label.

        Returns:
            ``torch.Tensor`` | ``None``: Validated ground truth label as a boolean
                tensor, or None.

        Raises:
            TypeError: If ``label`` is neither an integer nor a torch.Tensor.
            ValueError: If ``label`` shape or dtype is invalid.

        Example:
            >>> import torch
            >>> from anomalib.data.validators import ImageValidator
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

        This method validates and normalizes input masks. It handles:
            - Single channel masks only
            - [H, W] and [1, H, W] formats
            - Type conversion to boolean
            - None inputs return None

        Args:
            mask (``torch.Tensor`` | ``None``): Input ground truth mask.

        Returns:
            ``Mask`` | ``None``: Validated ground truth mask as a torchvision Mask
                object, or None.

        Raises:
            TypeError: If ``mask`` is not a torch.Tensor.
            ValueError: If ``mask`` dimensions or channels are invalid.

        Example:
            >>> import torch
            >>> from anomalib.data.validators import ImageValidator
            >>> mask = torch.randint(0, 2, (1, 224, 224))  # [1, H, W]
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

        This method validates and normalizes input anomaly maps. It handles:
            - Single channel maps only
            - [H, W] and [1, H, W] formats
            - Type conversion to float32
            - None inputs return None

        Args:
            anomaly_map (``torch.Tensor`` | ``None``): Input anomaly map.

        Returns:
            ``Mask`` | ``None``: Validated anomaly map as a torchvision Mask object,
                or None.

        Raises:
            TypeError: If ``anomaly_map`` is not a torch.Tensor.
            ValueError: If ``anomaly_map`` dimensions or channels are invalid.

        Example:
            >>> import torch
            >>> from anomalib.data.validators import ImageValidator
            >>> anomaly_map = torch.rand(1, 224, 224)  # [1, H, W]
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

        This method validates input image file paths.

        Args:
            image_path (``str`` | ``None``): Input image path to validate.

        Returns:
            ``str`` | ``None``: Validated image path, or None.

        Example:
            >>> from anomalib.data.validators import ImageValidator
            >>> path = "/path/to/image.jpg"
            >>> validated_path = ImageValidator.validate_image_path(path)
            >>> validated_path == path
            True
        """
        return validate_path(image_path) if image_path else None

    @staticmethod
    def validate_mask_path(mask_path: str | None) -> str | None:
        """Validate the mask path.

        This method validates input mask file paths.

        Args:
            mask_path (``str`` | ``None``): Input mask path to validate.

        Returns:
            ``str`` | ``None``: Validated mask path, or None.

        Example:
            >>> from anomalib.data.validators import ImageValidator
            >>> path = "/path/to/mask.png"
            >>> validated_path = ImageValidator.validate_mask_path(path)
            >>> validated_path == path
            True
        """
        return validate_path(mask_path) if mask_path else None

    @staticmethod
    def validate_pred_score(
        pred_score: torch.Tensor | np.ndarray | float | None,
    ) -> torch.Tensor | None:
        """Validate the prediction score.

        This method validates and normalizes prediction scores. It handles:
            - Float, numpy array and tensor inputs
            - Type conversion to float32
            - None inputs return None

        Args:
            pred_score (``torch.Tensor`` | ``np.ndarray`` | ``float`` | ``None``):
                Input prediction score.

        Returns:
            ``torch.Tensor`` | ``None``: Validated prediction score as a float32
                tensor, or None.

        Raises:
            TypeError: If ``pred_score`` cannot be converted to a tensor.
            ValueError: If ``pred_score`` is not a scalar.

        Example:
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
        """
        if pred_score is None:
            return None

        if not isinstance(pred_score, torch.Tensor):
            try:
                pred_score = torch.tensor(pred_score)
            except Exception as e:
                msg = "Failed to convert pred_score to a torch.Tensor."
                raise ValueError(msg) from e

        return pred_score.to(torch.float32)

    @staticmethod
    def validate_pred_mask(pred_mask: torch.Tensor | None) -> Mask | None:
        """Validate the prediction mask.

        This method validates and normalizes prediction masks. It handles:
            - Single channel masks only
            - [H, W] and [1, H, W] formats
            - Type conversion to boolean
            - None inputs return None

        Args:
            pred_mask (``torch.Tensor`` | ``None``): Input prediction mask.

        Returns:
            ``Mask`` | ``None``: Validated prediction mask as a torchvision Mask
                object, or None.

        Example:
            >>> import torch
            >>> from anomalib.data.validators import ImageValidator
            >>> mask = torch.randint(0, 2, (1, 224, 224))  # [1, H, W]
            >>> validated_mask = ImageValidator.validate_pred_mask(mask)
            >>> isinstance(validated_mask, Mask)
            True
            >>> validated_mask.shape
            torch.Size([224, 224])
        """
        return ImageValidator.validate_gt_mask(pred_mask)  # We can reuse the gt_mask validation

    @staticmethod
    def validate_pred_label(pred_label: torch.Tensor | np.ndarray | float | None) -> torch.Tensor | None:
        """Validate the prediction label.

        This method validates and normalizes prediction labels. It handles:
            - Float, numpy array and tensor inputs
            - Type conversion to boolean
            - None inputs return None

        Args:
            pred_label (``torch.Tensor`` | ``np.ndarray`` | ``float`` | ``None``):
                Input prediction label.

        Returns:
            ``torch.Tensor`` | ``None``: Validated prediction label as a boolean
                tensor, or None.

        Raises:
            TypeError: If ``pred_label`` cannot be converted to a tensor.
            ValueError: If ``pred_label`` is not a scalar.

        Example:
            >>> import torch
            >>> from anomalib.data.validators import ImageValidator
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

    @staticmethod
    def validate_explanation(explanation: str | None) -> str | None:
        """Validate the explanation string.

        This method validates explanation strings.

        Args:
            explanation (``str`` | ``None``): Input explanation string.

        Returns:
            ``str`` | ``None``: Validated explanation string, or None.

        Raises:
            TypeError: If ``explanation`` is not a string.

        Example:
            >>> from anomalib.data.validators import ImageValidator
            >>> explanation = "The image has a crack on the wall."
            >>> validated = ImageValidator.validate_explanation(explanation)
            >>> validated == explanation
            True
        """
        if explanation is None:
            return None
        if not isinstance(explanation, str):
            msg = f"Explanation must be a string, got {type(explanation)}."
            raise TypeError(msg)
        return explanation


class ImageBatchValidator:
    """Validate torch.Tensor data for batches of images.

    This class provides validation methods for batches of image data stored as PyTorch tensors.
    It ensures data consistency and correctness for images and associated metadata.

    The validator checks:
        - Tensor shapes and dimensions
        - Data types
        - Value ranges
        - Label formats
        - Mask properties
        - Path validity

    Example:
        Validate a batch of images and associated metadata::

            >>> from anomalib.data.validators import ImageBatchValidator
            >>> validator = ImageBatchValidator()
            >>> images = torch.rand(32, 3, 256, 256)  # [N, C, H, W]
            >>> labels = torch.zeros(32)
            >>> masks = torch.zeros((32, 256, 256))
            >>> validator.validate_image(images)
            >>> validator.validate_gt_label(labels)
            >>> validator.validate_gt_mask(masks)

    Note:
        The validator is used internally by the data modules to ensure data
        consistency before processing.
    """

    @staticmethod
    def validate_image(image: torch.Tensor) -> Image:
        """Validate the image for a batch.

        This method validates batches of images stored as PyTorch tensors. It handles:
            - Single images and batches
            - RGB images only
            - Channel-first format [N, C, H, W]
            - Type conversion to float32

        Args:
            image (``torch.Tensor``): Input image tensor to validate.

        Returns:
            ``Image``: Validated image as a torchvision Image object.

        Raises:
            TypeError: If ``image`` is not a torch.Tensor.
            ValueError: If ``image`` dimensions or channels are invalid.

        Example:
            Validate RGB batch::

                >>> import torch
                >>> from anomalib.data.validators import ImageBatchValidator
                >>> image = torch.rand(32, 3, 224, 224)  # [N, C, H, W]
                >>> validated = ImageBatchValidator.validate_image(image)
                >>> validated.shape
                torch.Size([32, 3, 224, 224])
        """
        if not isinstance(image, torch.Tensor):
            msg = f"Image must be a torch.Tensor, got {type(image)}."
            raise TypeError(msg)
        if image.ndim not in {3, 4}:
            msg = f"Image must have shape [C, H, W] or [N, C, H, W], got shape {image.shape}."
            raise ValueError(msg)
        if image.ndim == 3:
            image = image.unsqueeze(0)  # add batch dimension
        if image.shape[1] != 3:
            msg = f"Image must have 3 channels, got {image.shape[1]}."
            raise ValueError(msg)
        return Image(image, dtype=torch.float32)

    @staticmethod
    def validate_gt_label(gt_label: torch.Tensor | Sequence[int] | None) -> torch.Tensor | None:
        """Validate the ground truth label for a batch.

        This method validates batches of ground truth labels. It handles:
            - Conversion to torch.Tensor if needed
            - Type conversion to boolean
            - Shape validation

        Args:
            gt_label (``torch.Tensor`` | ``Sequence[int]`` | ``None``): Input ground truth
                label.

        Returns:
            ``torch.Tensor`` | ``None``: Validated ground truth label as a boolean tensor,
                or None.

        Raises:
            TypeError: If ``gt_label`` is not a sequence of integers or torch.Tensor.
            ValueError: If ``gt_label`` shape or data type is invalid.

        Example:
            Validate ground truth labels::

                >>> import torch
                >>> from anomalib.data.validators import ImageBatchValidator
                >>> gt_label = torch.tensor([0, 1, 1, 0])
                >>> validated = ImageBatchValidator.validate_gt_label(gt_label)
                >>> validated
                tensor([False,  True,  True, False])
        """
        if gt_label is None:
            return None
        if isinstance(gt_label, Sequence):
            gt_label = torch.tensor(gt_label)
        if not isinstance(gt_label, torch.Tensor):
            msg = f"Ground truth label must be a sequence of integers or a torch.Tensor, got {type(gt_label)}."
            raise TypeError(msg)
        if gt_label.ndim != 1:
            msg = f"Ground truth label must be a 1-dimensional vector, got shape {gt_label.shape}."
            raise ValueError(msg)
        if torch.is_floating_point(gt_label):
            msg = f"Ground truth label must be boolean or integer, got {gt_label}."
            raise ValueError(msg)
        return gt_label.bool()

    @staticmethod
    def validate_gt_mask(gt_mask: torch.Tensor | None) -> Mask | None:
        """Validate the ground truth mask for a batch.

        This method validates batches of ground truth masks. It handles:
            - Single masks and batches
            - Shape normalization
            - Type conversion to boolean

        Args:
            gt_mask (``torch.Tensor`` | ``None``): Input ground truth mask.

        Returns:
            ``Mask`` | ``None``: Validated ground truth mask as a torchvision Mask object,
                or None.

        Raises:
            TypeError: If ``gt_mask`` is not a torch.Tensor.
            ValueError: If ``gt_mask`` shape is invalid.

        Example:
            Validate ground truth masks::

                >>> import torch
                >>> from anomalib.data.validators import ImageBatchValidator
                >>> gt_mask = torch.randint(0, 2, (4, 224, 224))
                >>> validated = ImageBatchValidator.validate_gt_mask(gt_mask)
                >>> validated.shape
                torch.Size([4, 224, 224])
        """
        if gt_mask is None:
            return None
        if not isinstance(gt_mask, torch.Tensor):
            msg = f"Ground truth mask must be a torch.Tensor, got {type(gt_mask)}."
            raise TypeError(msg)
        if gt_mask.ndim not in {2, 3, 4}:
            msg = f"Ground truth mask must have shape [H, W] or [N, H, W] or [N, 1, H, W] got shape {gt_mask.shape}."
            raise ValueError(msg)
        if gt_mask.ndim == 2:
            gt_mask = gt_mask.unsqueeze(0)
        if gt_mask.ndim == 4:
            if gt_mask.shape[1] != 1:
                msg = f"Ground truth mask must have 1 channel, got {gt_mask.shape[1]}."
                raise ValueError(msg)
            gt_mask = gt_mask.squeeze(1)
        return Mask(gt_mask, dtype=torch.bool)

    @staticmethod
    def validate_mask_path(mask_path: Sequence[str] | None) -> list[str] | None:
        """Validate the mask paths for a batch.

        This method validates batches of mask file paths.

        Args:
            mask_path (``Sequence[str]`` | ``None``): Input sequence of mask paths.

        Returns:
            ``list[str]`` | ``None``: Validated list of mask paths, or None.

        Raises:
            TypeError: If ``mask_path`` is not a sequence of strings.

        Example:
            Validate mask paths::

                >>> from anomalib.data.validators import ImageBatchValidator
                >>> mask_paths = ["path/to/mask_1.png", "path/to/mask_2.png"]
                >>> validated = ImageBatchValidator.validate_mask_path(mask_paths)
                >>> validated
                ['path/to/mask_1.png', 'path/to/mask_2.png']
        """
        if mask_path is None:
            return None
        if not isinstance(mask_path, Sequence):
            msg = f"Mask path must be a sequence of paths or strings, got {type(mask_path)}."
            raise TypeError(msg)
        return [str(path) for path in mask_path]

    @staticmethod
    def validate_anomaly_map(anomaly_map: torch.Tensor | np.ndarray | None) -> Mask | None:
        """Validate the anomaly map for a batch.

        This method validates batches of anomaly maps. It handles:
            - Conversion from numpy arrays
            - Shape normalization
            - Type conversion to float32

        Args:
            anomaly_map (``torch.Tensor`` | ``np.ndarray`` | ``None``): Input anomaly map.

        Returns:
            ``Mask`` | ``None``: Validated anomaly map as a torchvision Mask object, or None.

        Raises:
            ValueError: If ``anomaly_map`` cannot be converted to tensor or has invalid shape.

        Example:
            Validate anomaly maps::

                >>> import torch
                >>> from anomalib.data.validators import ImageBatchValidator
                >>> anomaly_map = torch.rand(4, 224, 224)
                >>> validated = ImageBatchValidator.validate_anomaly_map(anomaly_map)
                >>> validated.shape
                torch.Size([4, 224, 224])
        """
        if anomaly_map is None:
            return None
        if not isinstance(anomaly_map, torch.Tensor):
            try:
                anomaly_map = torch.tensor(anomaly_map)
            except Exception as e:
                msg = "Failed to convert anomaly_map to a torch.Tensor."
                raise ValueError(msg) from e
        if anomaly_map.ndim not in {2, 3, 4}:
            msg = f"Anomaly map must have shape [H, W] or [N, H, W] or [N, 1, H, W], got shape {anomaly_map.shape}."
            raise ValueError(msg)
        if anomaly_map.ndim == 2:
            anomaly_map = anomaly_map.unsqueeze(0)
        if anomaly_map.ndim == 4:
            if anomaly_map.shape[1] != 1:
                msg = f"Anomaly map must have 1 channel, got {anomaly_map.shape[1]}."
                raise ValueError(msg)
            anomaly_map = anomaly_map.squeeze(1)
        return Mask(anomaly_map, dtype=torch.float32)

    @staticmethod
    def validate_pred_score(
        pred_score: torch.Tensor | np.ndarray | Sequence[float] | None,
    ) -> torch.Tensor | None:
        """Validate the prediction scores for a batch.

        This method validates batches of prediction scores. It handles:
            - Conversion from numpy arrays and sequences
            - Type conversion to float32

        Args:
            pred_score (``torch.Tensor`` | ``Sequence[float]`` | ``None``): Input prediction
                scores.

        Returns:
            ``torch.Tensor`` | ``None``: Validated prediction scores as float32 tensor,
                or None.

        Raises:
            TypeError: If ``pred_score`` is not a valid input type.
            ValueError: If ``pred_score`` cannot be converted to tensor.

        Example:
            Validate prediction scores::

                >>> import torch
                >>> from anomalib.data.validators import ImageBatchValidator
                >>> scores = [0.8, 0.7, 0.9]
                >>> validated = ImageBatchValidator.validate_pred_score(scores)
                >>> validated
                tensor([0.8000, 0.7000, 0.9000])
        """
        if pred_score is None:
            return None

        if isinstance(pred_score, Sequence):
            pred_score = torch.tensor(pred_score)
        if not isinstance(pred_score, torch.Tensor):
            try:
                pred_score = torch.tensor(pred_score)
            except Exception as e:
                msg = "Failed to convert pred_score to a torch.Tensor."
                raise ValueError(msg) from e

        return pred_score.to(torch.float32)

    @staticmethod
    def validate_pred_mask(pred_mask: torch.Tensor | None) -> Mask | None:
        """Validate the prediction mask for a batch.

        This method validates batches of prediction masks using the same logic as ground
        truth masks.

        Args:
            pred_mask (``torch.Tensor`` | ``None``): Input prediction mask.

        Returns:
            ``Mask`` | ``None``: Validated prediction mask as a torchvision Mask object,
                or None.

        Example:
            Validate prediction masks::

                >>> import torch
                >>> from anomalib.data.validators import ImageBatchValidator
                >>> pred_mask = torch.randint(0, 2, (4, 224, 224))
                >>> validated = ImageBatchValidator.validate_pred_mask(pred_mask)
                >>> validated.shape
                torch.Size([4, 224, 224])
        """
        return ImageBatchValidator.validate_gt_mask(pred_mask)  # We can reuse the gt_mask validation

    @staticmethod
    def validate_pred_label(pred_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction label for a batch.

        This method validates batches of prediction labels. It handles:
            - Shape normalization
            - Type conversion to boolean

        Args:
            pred_label (``torch.Tensor`` | ``None``): Input prediction label.

        Returns:
            ``torch.Tensor`` | ``None``: Validated prediction label as boolean tensor,
                or None.

        Raises:
            TypeError: If ``pred_label`` is not a torch.Tensor.
            ValueError: If ``pred_label`` has invalid shape.

        Example:
            Validate prediction labels::

                >>> import torch
                >>> from anomalib.data.validators import ImageBatchValidator
                >>> pred_label = torch.tensor([[1], [0], [1], [1]])
                >>> validated = ImageBatchValidator.validate_pred_label(pred_label)
                >>> validated
                tensor([ True, False,  True,  True])
        """
        if pred_label is None:
            return None
        if not isinstance(pred_label, torch.Tensor):
            msg = f"Predicted label must be a torch.Tensor, got {type(pred_label)}."
            raise TypeError(msg)
        if pred_label.ndim > 2:
            msg = f"Predicted label must be 1-dimensional or 2-dimensional, got shape {pred_label.shape}."
            raise ValueError(msg)
        if pred_label.ndim == 2:
            if pred_label.shape[0] == 1:
                pred_label = pred_label.squeeze(0)
            elif pred_label.shape[1] == 1:
                pred_label = pred_label.squeeze(1)
            else:
                msg = (
                    f"Predicted label with 2 dimensions must have shape [N, 1] or [1, N], got shape {pred_label.shape}."
                )
                raise ValueError(msg)
        return pred_label.to(torch.bool)

    @staticmethod
    def validate_image_path(image_path: list[str] | None) -> list[str] | None:
        """Validate the image paths for a batch.

        This method validates batches of image file paths.

        Args:
            image_path (``list[str]`` | ``None``): Input list of image paths.

        Returns:
            ``list[str]`` | ``None``: Validated list of image paths, or None.

        Raises:
            TypeError: If ``image_path`` is not a list of strings.

        Example:
            Validate image paths::

                >>> from anomalib.data.validators import ImageBatchValidator
                >>> image_paths = ["path/to/image_1.jpg", "path/to/image_2.jpg"]
                >>> validated = ImageBatchValidator.validate_image_path(image_paths)
                >>> validated
                ['path/to/image_1.jpg', 'path/to/image_2.jpg']
        """
        if image_path is None:
            return None
        if not isinstance(image_path, list):
            msg = f"Image path must be a list of strings, got {type(image_path)}."
            raise TypeError(msg)
        return [str(path) for path in image_path]

    @staticmethod
    def validate_explanation(explanation: list[str] | None) -> list[str] | None:
        """Validate the explanations for a batch.

        This method validates batches of explanation strings.

        Args:
            explanation (``list[str]`` | ``None``): Input list of explanations.

        Returns:
            ``list[str]`` | ``None``: Validated list of explanations, or None.

        Raises:
            TypeError: If ``explanation`` is not a list of strings.

        Example:
            Validate explanations::

                >>> from anomalib.data.validators import ImageBatchValidator
                >>> explanations = ["Crack on wall", "Dent on car"]
                >>> validated = ImageBatchValidator.validate_explanation(explanations)
                >>> validated
                ['Crack on wall', 'Dent on car']
        """
        if explanation is None:
            return None
        if not isinstance(explanation, list):
            msg = f"Explanation must be a list of strings, got {type(explanation)}."
            raise TypeError(msg)
        return [str(exp) for exp in explanation]
