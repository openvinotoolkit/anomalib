"""Validate numpy depth data.

This module provides validators for depth map data stored as numpy arrays. It includes
two main validator classes:

- ``NumpyDepthValidator``: Validates single depth map numpy arrays
- ``NumpyDepthBatchValidator``: Validates batches of depth map numpy arrays

The validators check that inputs meet format requirements like shape, type, etc.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np
from anomalib.data.validators.numpy.image import NumpyImageBatchValidator, NumpyImageValidator
from anomalib.data.validators.path import validate_path


class NumpyDepthValidator:
    """Validate numpy.ndarray data for depth images.

    This class provides methods to validate depth map data and associated metadata
    like labels, masks, etc. It ensures inputs meet format requirements.
    """

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Validate an image array.

        Args:
            image: Input image as numpy array

        Returns:
            Validated image array

        Example:
            >>> import numpy as np
            >>> validator = NumpyDepthValidator()
            >>> image = np.random.rand(32, 32, 3)
            >>> validated = validator.validate_image(image)
        """
        return NumpyImageValidator.validate_image(image)

    @staticmethod
    def validate_gt_label(label: int | np.ndarray | None) -> np.ndarray | None:
        """Validate a ground truth label.

        Args:
            label: Input label as integer or numpy array

        Returns:
            Validated label array or None
        """
        return NumpyImageValidator.validate_gt_label(label)

    @staticmethod
    def validate_gt_mask(mask: np.ndarray | None) -> np.ndarray | None:
        """Validate a ground truth mask.

        Args:
            mask: Input mask as numpy array

        Returns:
            Validated mask array or None
        """
        return NumpyImageValidator.validate_gt_mask(mask)

    @staticmethod
    def validate_mask_path(mask_path: str | None) -> str | None:
        """Validate a mask file path.

        Args:
            mask_path: Path to mask file

        Returns:
            Validated path string or None
        """
        return NumpyImageValidator.validate_mask_path(mask_path)

    @staticmethod
    def validate_anomaly_map(anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate an anomaly map.

        Args:
            anomaly_map: Input anomaly map as numpy array

        Returns:
            Validated anomaly map array or None
        """
        return NumpyImageValidator.validate_anomaly_map(anomaly_map)

    @staticmethod
    def validate_pred_score(
        pred_score: np.ndarray | float | None,
        anomaly_map: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Validate a prediction score.

        Args:
            pred_score: Prediction score as float or numpy array
            anomaly_map: Optional anomaly map for validation

        Returns:
            Validated prediction score array or None
        """
        return NumpyImageValidator.validate_pred_score(pred_score, anomaly_map)

    @staticmethod
    def validate_pred_mask(pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate a prediction mask.

        Args:
            pred_mask: Input prediction mask as numpy array

        Returns:
            Validated prediction mask array or None
        """
        return NumpyImageValidator.validate_pred_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate a prediction label.

        Args:
            pred_label: Input prediction label as numpy array

        Returns:
            Validated prediction label array or None
        """
        return NumpyImageValidator.validate_pred_label(pred_label)

    @staticmethod
    def validate_image_path(image_path: str | None) -> str | None:
        """Validate an image file path.

        Args:
            image_path: Path to image file

        Returns:
            Validated path string or None
        """
        return NumpyImageValidator.validate_image_path(image_path)

    @staticmethod
    def validate_depth_map(depth_map: np.ndarray | None) -> np.ndarray | None:
        """Validate a depth map array.

        Args:
            depth_map: Input depth map as numpy array

        Returns:
            Validated depth map array or None

        Raises:
            TypeError: If depth_map is not a numpy array
            ValueError: If depth_map has invalid shape

        Example:
            >>> import numpy as np
            >>> validator = NumpyDepthValidator()
            >>> depth = np.random.rand(32, 32)  # Valid 2D depth map
            >>> validated = validator.validate_depth_map(depth)
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
        """Validate a depth map file path.

        Args:
            depth_path: Path to depth map file

        Returns:
            Validated path string or None
        """
        return validate_path(depth_path) if depth_path else None

    @staticmethod
    def validate_explanation(explanation: str | None) -> str | None:
        """Validate an explanation string.

        Args:
            explanation: Input explanation string

        Returns:
            Validated explanation string or None
        """
        return NumpyImageValidator.validate_explanation(explanation)


class NumpyDepthBatchValidator:
    """Validate numpy.ndarray data for batches of depth images.

    This class provides methods to validate batches of depth map data and
    associated metadata. It ensures inputs meet format requirements.
    """

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Validate a batch of images.

        Args:
            image: Input image batch as numpy array

        Returns:
            Validated image batch array
        """
        return NumpyImageBatchValidator.validate_image(image)

    @staticmethod
    def validate_gt_label(
        gt_label: np.ndarray | Sequence[int] | None,
    ) -> np.ndarray | None:
        """Validate ground truth labels for a batch.

        Args:
            gt_label: Input labels as array or sequence

        Returns:
            Validated label batch array or None
        """
        return NumpyImageBatchValidator.validate_gt_label(gt_label)

    @staticmethod
    def validate_gt_mask(gt_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate ground truth masks for a batch.

        Args:
            gt_mask: Input mask batch as numpy array

        Returns:
            Validated mask batch array or None
        """
        return NumpyImageBatchValidator.validate_gt_mask(gt_mask)

    @staticmethod
    def validate_mask_path(mask_path: Sequence[str] | None) -> list[str] | None:
        """Validate mask file paths for a batch.

        Args:
            mask_path: Sequence of paths to mask files

        Returns:
            List of validated path strings or None
        """
        return NumpyImageBatchValidator.validate_mask_path(mask_path)

    @staticmethod
    def validate_anomaly_map(anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate anomaly maps for a batch.

        Args:
            anomaly_map: Input anomaly map batch as numpy array

        Returns:
            Validated anomaly map batch array or None
        """
        return NumpyImageBatchValidator.validate_anomaly_map(anomaly_map)

    @staticmethod
    def validate_pred_score(pred_score: np.ndarray | None) -> np.ndarray | None:
        """Validate prediction scores for a batch.

        Args:
            pred_score: Input prediction scores as numpy array

        Returns:
            Validated prediction score batch array or None
        """
        return NumpyImageBatchValidator.validate_pred_score(pred_score)

    @staticmethod
    def validate_pred_mask(pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate prediction masks for a batch.

        Args:
            pred_mask: Input prediction mask batch as numpy array

        Returns:
            Validated prediction mask batch array or None
        """
        return NumpyImageBatchValidator.validate_pred_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate prediction labels for a batch.

        Args:
            pred_label: Input prediction label batch as numpy array

        Returns:
            Validated prediction label batch array or None
        """
        return NumpyImageBatchValidator.validate_pred_label(pred_label)

    @staticmethod
    def validate_image_path(image_path: list[str] | None) -> list[str] | None:
        """Validate image file paths for a batch.

        Args:
            image_path: List of paths to image files

        Returns:
            List of validated path strings or None
        """
        return NumpyImageBatchValidator.validate_image_path(image_path)

    @staticmethod
    def validate_depth_map(depth_map: np.ndarray | None) -> np.ndarray | None:
        """Validate depth maps for a batch.

        Args:
            depth_map: Input depth map batch as numpy array

        Returns:
            Validated depth map batch array or None

        Raises:
            TypeError: If depth_map is not a numpy array
            ValueError: If depth_map has invalid shape

        Example:
            >>> import numpy as np
            >>> validator = NumpyDepthBatchValidator()
            >>> depth = np.random.rand(4, 32, 32)  # Valid batch of 2D depth maps
            >>> validated = validator.validate_depth_map(depth)
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
            depth_path: List of paths to depth map files

        Returns:
            List of validated path strings or None

        Raises:
            TypeError: If depth_path is not a list
        """
        if depth_path is None:
            return None
        if not isinstance(depth_path, list):
            msg = f"Depth path must be a list of strings, got {type(depth_path)}."
            raise TypeError(msg)
        return [validate_path(path) for path in depth_path]

    @staticmethod
    def validate_explanation(explanation: list[str] | None) -> list[str] | None:
        """Validate explanations for a batch.

        Args:
            explanation: List of explanation strings

        Returns:
            List of validated explanation strings or None
        """
        return NumpyImageBatchValidator.validate_explanation(explanation)
