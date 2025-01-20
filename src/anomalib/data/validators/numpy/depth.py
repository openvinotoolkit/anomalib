"""Validate numpy depth data.

This module provides validators for depth data stored as numpy arrays. The validators
ensure data consistency and correctness for depth maps and batches of depth maps.

The validators check:
    - Array shapes and dimensions
    - Data types
    - Value ranges
    - Label formats
    - Mask properties

Example:
    Validate a single depth map::

        >>> from anomalib.data.validators import NumpyDepthValidator
        >>> validator = NumpyDepthValidator()
        >>> validator.validate_image(depth_map)

    Validate a batch of depth maps::

        >>> from anomalib.data.validators import NumpyDepthBatchValidator
        >>> validator = NumpyDepthBatchValidator()
        >>> validator(depth_maps=depth_maps, labels=labels, masks=masks)

Note:
    The validators are used internally by the data modules to ensure data
    consistency before processing depth map data.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np
from anomalib.data.validators.numpy.image import NumpyImageBatchValidator, NumpyImageValidator
from anomalib.data.validators.path import validate_path


class NumpyDepthValidator:
    """Validate numpy depth data.

    This class provides validation methods for depth data stored as numpy arrays.
    It ensures data consistency and correctness for depth maps and associated
    metadata.

    The validator checks:
        - Array shapes and dimensions
        - Data types
        - Value ranges
        - Label formats
        - Mask properties
        - Path validity

    Example:
        Validate a depth map and associated metadata::

            >>> from anomalib.data.validators import NumpyDepthValidator
            >>> validator = NumpyDepthValidator()
            >>> depth_map = np.random.rand(256, 256).astype(np.float32)
            >>> validated_map = validator.validate_depth_map(depth_map)
    """

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Validate image array.

        Args:
            image (np.ndarray): Input image to validate.

        Returns:
            np.ndarray: Validated image array.
        """
        return NumpyImageValidator.validate_image(image)

    @staticmethod
    def validate_gt_label(label: int | np.ndarray | None) -> np.ndarray | None:
        """Validate ground truth label.

        Args:
            label (int | np.ndarray | None): Input label to validate.

        Returns:
            np.ndarray | None: Validated label.
        """
        return NumpyImageValidator.validate_gt_label(label)

    @staticmethod
    def validate_gt_mask(mask: np.ndarray | None) -> np.ndarray | None:
        """Validate ground truth mask.

        Args:
            mask (np.ndarray | None): Input mask to validate.

        Returns:
            np.ndarray | None: Validated mask.
        """
        return NumpyImageValidator.validate_gt_mask(mask)

    @staticmethod
    def validate_mask_path(mask_path: str | None) -> str | None:
        """Validate mask path.

        Args:
            mask_path (str | None): Path to mask file.

        Returns:
            str | None: Validated mask path.
        """
        return NumpyImageValidator.validate_mask_path(mask_path)

    @staticmethod
    def validate_anomaly_map(anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate anomaly map.

        Args:
            anomaly_map (np.ndarray | None): Input anomaly map to validate.

        Returns:
            np.ndarray | None: Validated anomaly map.
        """
        return NumpyImageValidator.validate_anomaly_map(anomaly_map)

    @staticmethod
    def validate_pred_score(
        pred_score: np.ndarray | float | None,
        anomaly_map: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Validate prediction score.

        Args:
            pred_score (np.ndarray | float | None): Input prediction score.
            anomaly_map (np.ndarray | None, optional): Associated anomaly map.
                Defaults to None.

        Returns:
            np.ndarray | None: Validated prediction score.
        """
        return NumpyImageValidator.validate_pred_score(pred_score, anomaly_map)

    @staticmethod
    def validate_pred_mask(pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate prediction mask.

        Args:
            pred_mask (np.ndarray | None): Input prediction mask to validate.

        Returns:
            np.ndarray | None: Validated prediction mask.
        """
        return NumpyImageValidator.validate_pred_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate prediction label.

        Args:
            pred_label (np.ndarray | None): Input prediction label to validate.

        Returns:
            np.ndarray | None: Validated prediction label.
        """
        return NumpyImageValidator.validate_pred_label(pred_label)

    @staticmethod
    def validate_image_path(image_path: str | None) -> str | None:
        """Validate image path.

        Args:
            image_path (str | None): Path to image file.

        Returns:
            str | None: Validated image path.
        """
        return NumpyImageValidator.validate_image_path(image_path)

    @staticmethod
    def validate_depth_map(depth_map: np.ndarray | None) -> np.ndarray | None:
        """Validate depth map array.

        Ensures the depth map has correct dimensions and data type.

        Args:
            depth_map (np.ndarray | None): Input depth map to validate.

        Returns:
            np.ndarray | None: Validated depth map as float32.

        Raises:
            TypeError: If depth map is not a numpy array.
            ValueError: If depth map dimensions are invalid.

        Example:
            >>> depth_map = np.random.rand(256, 256).astype(np.float32)
            >>> validated = NumpyDepthValidator.validate_depth_map(depth_map)
            >>> validated.shape
            (256, 256)
        """
        if depth_map is None:
            return None
        if not isinstance(depth_map, np.ndarray):
            msg = f"Depth map must be a numpy array, got {type(depth_map)}."
            raise TypeError(msg)
        if depth_map.ndim not in {2, 3}:
            msg = f"Depth map must have shape [H, W] or [H, W, 1], got shape {depth_map.shape}."
            raise ValueError(msg)
        if depth_map.ndim == 3 and depth_map.shape[2] != 1:
            msg = f"Depth map with 3 dimensions must have 1 channel, got {depth_map.shape[2]}."
            raise ValueError(msg)
        return depth_map.astype(np.float32)

    @staticmethod
    def validate_depth_path(depth_path: str | None) -> str | None:
        """Validate depth map file path.

        Args:
            depth_path (str | None): Path to depth map file.

        Returns:
            str | None: Validated depth map path.
        """
        return validate_path(depth_path) if depth_path else None

    @staticmethod
    def validate_explanation(explanation: str | None) -> str | None:
        """Validate explanation string.

        Args:
            explanation (str | None): Input explanation to validate.

        Returns:
            str | None: Validated explanation string.
        """
        return NumpyImageValidator.validate_explanation(explanation)


class NumpyDepthBatchValidator:
    """Validate numpy depth data batches.

    This class provides validation methods for batches of depth data stored as numpy arrays.
    It ensures data consistency and correctness for batches of depth maps and associated
    metadata.

    The validator checks:
        - Array shapes and dimensions
        - Data types
        - Value ranges
        - Label formats
        - Mask properties
        - Path validity

    Example:
        Validate a batch of depth maps and associated metadata::

            >>> from anomalib.data.validators import NumpyDepthBatchValidator
            >>> validator = NumpyDepthBatchValidator()
            >>> depth_maps = np.random.rand(32, 256, 256).astype(np.float32)
            >>> labels = np.zeros(32)
            >>> masks = np.zeros((32, 256, 256))
            >>> validator.validate_depth_map(depth_maps)
            >>> validator.validate_gt_label(labels)
            >>> validator.validate_gt_mask(masks)
    """

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Validate image batch array.

        Args:
            image (np.ndarray): Input image batch to validate.

        Returns:
            np.ndarray: Validated image batch array.
        """
        return NumpyImageBatchValidator.validate_image(image)

    @staticmethod
    def validate_gt_label(gt_label: np.ndarray | Sequence[int] | None) -> np.ndarray | None:
        """Validate ground truth label batch.

        Args:
            gt_label (np.ndarray | Sequence[int] | None): Input label batch to validate.

        Returns:
            np.ndarray | None: Validated label batch.
        """
        return NumpyImageBatchValidator.validate_gt_label(gt_label)

    @staticmethod
    def validate_gt_mask(gt_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate ground truth mask batch.

        Args:
            gt_mask (np.ndarray | None): Input mask batch to validate.

        Returns:
            np.ndarray | None: Validated mask batch.
        """
        return NumpyImageBatchValidator.validate_gt_mask(gt_mask)

    @staticmethod
    def validate_mask_path(mask_path: Sequence[str] | None) -> list[str] | None:
        """Validate mask file paths for a batch.

        Args:
            mask_path (Sequence[str] | None): Sequence of mask file paths to validate.

        Returns:
            list[str] | None: Validated mask file paths.
        """
        return NumpyImageBatchValidator.validate_mask_path(mask_path)

    @staticmethod
    def validate_anomaly_map(anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate anomaly map batch.

        Args:
            anomaly_map (np.ndarray | None): Input anomaly map batch to validate.

        Returns:
            np.ndarray | None: Validated anomaly map batch.
        """
        return NumpyImageBatchValidator.validate_anomaly_map(anomaly_map)

    @staticmethod
    def validate_pred_score(pred_score: np.ndarray | None) -> np.ndarray | None:
        """Validate prediction scores for a batch.

        Args:
            pred_score (np.ndarray | None): Input prediction scores to validate.

        Returns:
            np.ndarray | None: Validated prediction scores.
        """
        return NumpyImageBatchValidator.validate_pred_score(pred_score)

    @staticmethod
    def validate_pred_mask(pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate prediction mask batch.

        Args:
            pred_mask (np.ndarray | None): Input prediction mask batch to validate.

        Returns:
            np.ndarray | None: Validated prediction mask batch.
        """
        return NumpyImageBatchValidator.validate_pred_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate prediction label batch.

        Args:
            pred_label (np.ndarray | None): Input prediction label batch to validate.

        Returns:
            np.ndarray | None: Validated prediction label batch.
        """
        return NumpyImageBatchValidator.validate_pred_label(pred_label)

    @staticmethod
    def validate_image_path(image_path: list[str] | None) -> list[str] | None:
        """Validate image file paths for a batch.

        Args:
            image_path (list[str] | None): List of image file paths to validate.

        Returns:
            list[str] | None: Validated image file paths.
        """
        return NumpyImageBatchValidator.validate_image_path(image_path)

    @staticmethod
    def validate_depth_map(depth_map: np.ndarray | None) -> np.ndarray | None:
        """Validate depth map batch.

        Args:
            depth_map (np.ndarray | None): Input depth map batch to validate.

        Returns:
            np.ndarray | None: Validated depth map batch as float32.

        Raises:
            TypeError: If depth map batch is not a numpy array.
            ValueError: If depth map batch dimensions are invalid.

        Example:
            >>> depth_maps = np.random.rand(32, 256, 256).astype(np.float32)
            >>> validated = NumpyDepthBatchValidator.validate_depth_map(depth_maps)
            >>> validated.shape
            (32, 256, 256)
        """
        if depth_map is None:
            return None
        if not isinstance(depth_map, np.ndarray):
            msg = f"Depth map batch must be a numpy array, got {type(depth_map)}."
            raise TypeError(msg)
        if depth_map.ndim not in {3, 4}:
            msg = f"Depth map batch must have shape [N, H, W] or [N, H, W, 1], got shape {depth_map.shape}."
            raise ValueError(msg)
        if depth_map.ndim == 4 and depth_map.shape[3] != 1:
            msg = f"Depth map batch with 4 dimensions must have 1 channel, got {depth_map.shape[3]}."
            raise ValueError(msg)
        return depth_map.astype(np.float32)

    @staticmethod
    def validate_depth_path(depth_path: list[str] | None) -> list[str] | None:
        """Validate depth map file paths for a batch.

        Args:
            depth_path (list[str] | None): List of depth map file paths to validate.

        Returns:
            list[str] | None: Validated depth map file paths.

        Raises:
            TypeError: If depth_path is not a list of strings.
        """
        if depth_path is None:
            return None
        if not isinstance(depth_path, list):
            msg = f"Depth path must be a list of strings, got {type(depth_path)}."
            raise TypeError(msg)
        return [validate_path(path) for path in depth_path]

    @staticmethod
    def validate_explanation(explanation: list[str] | None) -> list[str] | None:
        """Validate explanation strings for a batch.

        Args:
            explanation (list[str] | None): List of explanation strings to validate.

        Returns:
            list[str] | None: Validated explanation strings.
        """
        return NumpyImageBatchValidator.validate_explanation(explanation)
