"""Validate numpy video data.

This module provides validators for video data stored as numpy arrays. The validators
ensure data consistency and correctness for videos and batches of videos.

The validators check:
    - Array shapes and dimensions
    - Data types
    - Value ranges
    - Label formats
    - Mask properties

Example:
    Validate a single video::

        >>> from anomalib.data.validators import NumpyVideoValidator
        >>> validator = NumpyVideoValidator()
        >>> validator.validate_image(video)

    Validate a batch of videos::

        >>> from anomalib.data.validators import NumpyVideoBatchValidator
        >>> validator = NumpyVideoBatchValidator()
        >>> validator(videos=videos, labels=labels, masks=masks)

Note:
    The validators are used internally by the data modules to ensure data
    consistency before processing video data.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np
from anomalib.data.validators.numpy.image import NumpyImageBatchValidator, NumpyImageValidator
from anomalib.data.validators.path import validate_batch_path, validate_path


class NumpyVideoValidator:
    """Validate numpy array data for videos.

    This class provides validation methods for video data stored as numpy arrays.
    It ensures data consistency and correctness for videos and associated metadata.

    The validator checks:
        - Array shapes and dimensions
        - Data types
        - Value ranges
        - Label formats
        - Mask properties
        - Path validity

    Example:
        Validate a video and associated metadata::

            >>> from anomalib.data.validators import NumpyVideoValidator
            >>> validator = NumpyVideoValidator()
            >>> video = np.random.rand(10, 224, 224, 3)  # [T, H, W, C]
            >>> validated_video = validator.validate_image(video)
            >>> label = 1
            >>> validated_label = validator.validate_gt_label(label)
            >>> mask = np.random.randint(0, 2, (10, 224, 224))  # [T, H, W]
            >>> validated_mask = validator.validate_gt_mask(mask)

    Note:
        The validator is used internally by the data modules to ensure data
        consistency before processing.
    """

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Validate the video array.

        Validates and normalizes input video arrays. Handles both RGB and grayscale
        videos, and ensures proper time dimension.

        Args:
            image (``np.ndarray``): Input video array to validate.

        Returns:
            ``np.ndarray``: Validated video array in format [T, H, W, C] as float32.

        Raises:
            TypeError: If ``image`` is not a numpy array.
            ValueError: If ``image`` dimensions or channels are invalid.

        Example:
            Validate RGB video::

                >>> import numpy as np
                >>> validator = NumpyVideoValidator()
                >>> video = np.random.rand(10, 224, 224, 3)  # [T, H, W, C]
                >>> validated_video = validator.validate_image(video)
                >>> print(validated_video.shape, validated_video.dtype)
                (10, 224, 224, 3) float32
        """
        if not isinstance(image, np.ndarray):
            msg = f"Video must be a numpy.ndarray, got {type(image)}."
            raise TypeError(msg)

        if image.ndim == 3:
            # Add time dimension for single frame
            image = np.expand_dims(image, axis=0)

        if image.ndim != 4:
            msg = f"Video must have 4 dimensions [T, H, W, C], got shape {image.shape}."
            raise ValueError(msg)

        if image.shape[3] not in {1, 3}:
            msg = f"Video must have 1 or 3 channels, got {image.shape[3]}."
            raise ValueError(msg)

        return image.astype(np.float32)

    @staticmethod
    def validate_gt_label(label: int | np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth label.

        Args:
            label (``int`` | ``np.ndarray`` | ``None``): Input label to validate.

        Returns:
            ``np.ndarray`` | ``None``: Validated label as boolean numpy array, or None if
                input is None.

        Raises:
            TypeError: If ``label`` is not an integer or numpy array.
            ValueError: If ``label`` is not a scalar.

        Example:
            >>> validator = NumpyVideoValidator()
            >>> label = 1
            >>> validated_label = validator.validate_gt_label(label)
            >>> print(validated_label, validated_label.dtype)
            True bool
        """
        if label is None:
            return None
        if isinstance(label, int):
            label = np.array(label)
        if not isinstance(label, np.ndarray):
            msg = f"Ground truth label must be an integer or a numpy.ndarray, got {type(label)}."
            raise TypeError(msg)
        if label.ndim != 0:
            msg = f"Ground truth label must be a scalar, got shape {label.shape}."
            raise ValueError(msg)
        if not np.issubdtype(label.dtype, np.integer):
            msg = f"Ground truth label must be boolean or integer, got {label.dtype}."
            raise TypeError(msg)
        return label.astype(bool)

    @staticmethod
    def validate_gt_mask(mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth mask.

        Args:
            mask (``np.ndarray`` | ``None``): Input mask to validate.

        Returns:
            ``np.ndarray`` | ``None``: Validated mask as boolean numpy array, or None if
                input is None.

        Raises:
            TypeError: If ``mask`` is not a numpy array.
            ValueError: If ``mask`` dimensions or channel count are invalid.

        Example:
            >>> import numpy as np
            >>> validator = NumpyVideoValidator()
            >>> mask = np.random.randint(0, 2, size=(5, 224, 224))  # [T, H, W]
            >>> validated_mask = validator.validate_gt_mask(mask)
            >>> print(validated_mask.shape, validated_mask.dtype)
            (5, 224, 224) bool
        """
        if mask is None:
            return None
        if not isinstance(mask, np.ndarray):
            msg = f"Mask must be a numpy.ndarray, got {type(mask)}."
            raise TypeError(msg)
        if mask.ndim not in {3, 4}:
            msg = f"Mask must have shape [T, H, W] or [T, H, W, 1] got shape {mask.shape}."
            raise ValueError(msg)
        if mask.ndim == 4 and mask.shape[3] != 1:
            msg = f"Mask must have 1 channel, got {mask.shape[3]}."
            raise ValueError(msg)
        return mask.astype(bool)

    @staticmethod
    def validate_mask_path(mask_path: str | None) -> str | None:
        """Validate the mask path.

        Args:
            mask_path (``str`` | ``None``): Input mask path to validate.

        Returns:
            ``str`` | ``None``: Validated mask path, or None if input is None.

        Example:
            >>> validator = NumpyVideoValidator()
            >>> path = "/path/to/mask.png"
            >>> validated_path = validator.validate_mask_path(path)
            >>> print(validated_path)
            /path/to/mask.png
        """
        return validate_path(mask_path) if mask_path else None

    @staticmethod
    def validate_anomaly_map(anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate the anomaly map.

        Args:
            anomaly_map (``np.ndarray`` | ``None``): Input anomaly map to validate.

        Returns:
            ``np.ndarray`` | ``None``: Validated anomaly map as float32 numpy array, or
                None if input is None.

        Raises:
            TypeError: If ``anomaly_map`` is not a numpy array.
            ValueError: If ``anomaly_map`` dimensions or channel count are invalid.

        Example:
            >>> import numpy as np
            >>> validator = NumpyVideoValidator()
            >>> amap = np.random.rand(5, 224, 224)  # [T, H, W]
            >>> validated_amap = validator.validate_anomaly_map(amap)
            >>> print(validated_amap.shape, validated_amap.dtype)
            (5, 224, 224) float32
        """
        if anomaly_map is None:
            return None
        if not isinstance(anomaly_map, np.ndarray):
            msg = f"Anomaly map must be a numpy.ndarray, got {type(anomaly_map)}."
            raise TypeError(msg)
        if anomaly_map.ndim not in {3, 4}:
            msg = f"Anomaly map must have shape [T, H, W] or [T, H, W, 1], got shape {anomaly_map.shape}."
            raise ValueError(msg)
        if anomaly_map.ndim == 4 and anomaly_map.shape[3] != 1:
            msg = f"Anomaly map must have 1 channel, got {anomaly_map.shape[3]}."
            raise ValueError(msg)
        return anomaly_map.astype(np.float32)

    @staticmethod
    def validate_pred_score(pred_score: np.ndarray | float | None) -> np.ndarray | None:
        """Validate the prediction score.

        Args:
            pred_score (``np.ndarray`` | ``float`` | ``None``): Input prediction score to
                validate.

        Returns:
            ``np.ndarray`` | ``None``: Validated prediction score as float32 numpy array,
                or None if input is None.

        Raises:
            TypeError: If ``pred_score`` is not a float or numpy array.
            ValueError: If ``pred_score`` is not a scalar.

        Example:
            >>> validator = NumpyVideoValidator()
            >>> score = 0.75
            >>> validated_score = validator.validate_pred_score(score)
            >>> print(validated_score, validated_score.dtype)
            0.75 float32
        """
        if pred_score is None:
            return None
        if isinstance(pred_score, float):
            pred_score = np.array(pred_score)
        if not isinstance(pred_score, np.ndarray):
            msg = f"Prediction score must be a float or numpy.ndarray, got {type(pred_score)}."
            raise TypeError(msg)
        if pred_score.ndim != 0:
            msg = f"Prediction score must be a scalar, got shape {pred_score.shape}."
            raise ValueError(msg)
        return pred_score.astype(np.float32)

    @staticmethod
    def validate_pred_mask(pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction mask.

        Args:
            pred_mask (``np.ndarray`` | ``None``): Input prediction mask to validate.

        Returns:
            ``np.ndarray`` | ``None``: Validated prediction mask as boolean numpy array,
                or None if input is None.

        Example:
            >>> import numpy as np
            >>> validator = NumpyVideoValidator()
            >>> mask = np.random.randint(0, 2, size=(5, 224, 224))  # [T, H, W]
            >>> validated_mask = validator.validate_pred_mask(mask)
            >>> print(validated_mask.shape, validated_mask.dtype)
            (5, 224, 224) bool
        """
        return NumpyVideoValidator.validate_gt_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction label.

        Args:
            pred_label (``np.ndarray`` | ``None``): Input prediction label to validate.

        Returns:
            ``np.ndarray`` | ``None``: Validated prediction label as boolean numpy array,
                or None if input is None.

        Raises:
            ValueError: If ``pred_label`` cannot be converted to a numpy array or is not
                a scalar.

        Example:
            >>> import numpy as np
            >>> validator = NumpyVideoValidator()
            >>> label = np.array(1)
            >>> validated_label = validator.validate_pred_label(label)
            >>> print(validated_label, validated_label.dtype)
            True bool
        """
        if pred_label is None:
            return None
        if not isinstance(pred_label, np.ndarray):
            try:
                pred_label = np.array(pred_label)
            except Exception as e:
                msg = "Failed to convert pred_label to a numpy.ndarray."
                raise ValueError(msg) from e
        pred_label = pred_label.squeeze()
        if pred_label.ndim != 0:
            msg = f"Predicted label must be a scalar, got shape {pred_label.shape}."
            raise ValueError(msg)
        return pred_label.astype(bool)

    @staticmethod
    def validate_video_path(video_path: str | None) -> str | None:
        """Validate the video path.

        Args:
            video_path (``str`` | ``None``): Input video path to validate.

        Returns:
            ``str`` | ``None``: Validated video path, or None if input is None.

        Example:
            >>> validator = NumpyVideoValidator()
            >>> path = "/path/to/video.mp4"
            >>> validated_path = validator.validate_video_path(path)
            >>> print(validated_path)
            /path/to/video.mp4
        """
        return validate_path(video_path) if video_path else None

    @staticmethod
    def validate_original_image(original_image: np.ndarray | None) -> np.ndarray | None:
        """Validate the original video.

        Args:
            original_image (``np.ndarray`` | ``None``): Input original video to validate.

        Returns:
            ``np.ndarray`` | ``None``: Validated original video, or None if input is None.

        Raises:
            TypeError: If ``original_image`` is not a numpy array.
            ValueError: If ``original_image`` dimensions or channel count are invalid.

        Example:
            >>> import numpy as np
            >>> validator = NumpyVideoValidator()
            >>> video = np.random.rand(10, 224, 224, 3)  # [T, H, W, C]
            >>> validated_video = validator.validate_original_image(video)
            >>> print(validated_video.shape, validated_video.dtype)
            (10, 224, 224, 3) float64
        """
        if original_image is None:
            return None
        if not isinstance(original_image, np.ndarray):
            msg = f"Original image must be a numpy.ndarray, got {type(original_image)}."
            raise TypeError(msg)
        if original_image.ndim not in {3, 4}:
            msg = f"Original image must have shape [T, H, W, C] or [H, W, C], got shape {original_image.shape}."
            raise ValueError(msg)
        if original_image.shape[-1] != 3:
            msg = f"Original image must have 3 channels, got {original_image.shape[-1]}."
            raise ValueError(msg)
        return original_image

    @staticmethod
    def validate_target_frame(target_frame: int | None) -> int | None:
        """Validate the target frame index.

        Args:
            target_frame (``int`` | ``None``): Input target frame index to validate.

        Returns:
            ``int`` | ``None``: Validated target frame index, or None if input is None.

        Raises:
            TypeError: If ``target_frame`` is not an integer.
            ValueError: If ``target_frame`` is negative.

        Example:
            >>> validator = NumpyVideoValidator()
            >>> frame = 5
            >>> validated_frame = validator.validate_target_frame(frame)
            >>> print(validated_frame)
            5
        """
        if target_frame is None:
            return None
        if not isinstance(target_frame, int):
            msg = f"Target frame must be an integer, got {type(target_frame)}."
            raise TypeError(msg)
        if target_frame < 0:
            msg = "Target frame index must be non-negative."
            raise ValueError(msg)
        return target_frame

    @staticmethod
    def validate_explanation(explanation: str | None) -> str | None:
        """Validate the explanation string."""
        return NumpyImageValidator.validate_explanation(explanation)


class NumpyVideoBatchValidator:
    """Validate numpy array data for batches of videos.

    This class provides validation methods for batches of video data stored as numpy arrays.
    It ensures data consistency and correctness for video batches and associated metadata.

    The validator checks:
        - Array shapes and dimensions
        - Data types
        - Value ranges
        - Label formats
        - Mask properties
        - Path validity

    Example:
        Validate a batch of videos and associated metadata::

            >>> from anomalib.data.validators import NumpyVideoBatchValidator
            >>> validator = NumpyVideoBatchValidator()
            >>> videos = np.random.rand(32, 10, 224, 224, 3)  # [N, T, H, W, C]
            >>> labels = np.zeros(32)
            >>> masks = np.zeros((32, 10, 224, 224))
            >>> validated_videos = validator.validate_image(videos)
            >>> validated_labels = validator.validate_gt_label(labels)
            >>> validated_masks = validator.validate_gt_mask(masks)

    Note:
        The validator is used internally by the data modules to ensure data
        consistency before processing.
    """

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Validate the video batch array.

        Args:
            image (``np.ndarray``): Input video batch array to validate.

        Returns:
            ``np.ndarray``: Validated video batch array as float32.

        Raises:
            TypeError: If the input is not a numpy array.
            ValueError: If the array dimensions or channel count are invalid.

        Example:
            >>> import numpy as np
            >>> validator = NumpyVideoBatchValidator()
            >>> video_batch = np.random.rand(2, 10, 224, 224, 3)  # [N, T, H, W, C]
            >>> validated_batch = validator.validate_image(video_batch)
            >>> print(validated_batch.shape, validated_batch.dtype)
            (2, 10, 224, 224, 3) float32
        """
        if not isinstance(image, np.ndarray):
            msg = f"Video batch must be a numpy.ndarray, got {type(image)}."
            raise TypeError(msg)
        if image.ndim not in {4, 5}:
            msg = f"Video batch must have 4 or 5 dimensions, got shape {image.shape}."
            raise ValueError(msg)
        if image.ndim == 4:
            if image.shape[3] not in {1, 3}:
                msg = f"Video batch must have 1 or 3 channels for single frame, got {image.shape[3]}."
                raise ValueError(msg)
        elif image.ndim == 5 and image.shape[4] not in {1, 3}:
            msg = f"Video batch must have 1 or 3 channels, got {image.shape[4]}."
            raise ValueError(msg)
        return image.astype(np.float32)

    @staticmethod
    def validate_gt_label(gt_label: np.ndarray | Sequence[int] | None) -> np.ndarray | None:
        """Validate the ground truth label batch.

        Args:
            gt_label (``np.ndarray`` | ``Sequence[int]`` | ``None``): Input ground truth
                label batch to validate.

        Returns:
            ``np.ndarray`` | ``None``: Validated ground truth label batch as boolean numpy
                array, or None if input is None.

        Raises:
            TypeError: If the input is not a numpy array or sequence of integers.
            ValueError: If the label batch shape is invalid.

        Example:
            >>> import numpy as np
            >>> validator = NumpyVideoBatchValidator()
            >>> labels = [0, 1, 1, 0]
            >>> validated_labels = validator.validate_gt_label(labels)
            >>> print(validated_labels, validated_labels.dtype)
            [False  True  True False] bool
        """
        if gt_label is None:
            return None
        if isinstance(gt_label, Sequence):
            gt_label = np.array(gt_label)
        if not isinstance(gt_label, np.ndarray):
            msg = f"Ground truth label batch must be a numpy.ndarray, got {type(gt_label)}."
            raise TypeError(msg)
        if gt_label.ndim != 1:
            msg = f"Ground truth label batch must be 1-dimensional, got shape {gt_label.shape}."
            raise ValueError(msg)
        return gt_label.astype(bool)

    @staticmethod
    def validate_gt_mask(gt_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth mask batch.

        Args:
            gt_mask (``np.ndarray`` | ``None``): Input ground truth mask batch to validate.

        Returns:
            ``np.ndarray`` | ``None``: Validated ground truth mask batch as boolean numpy
                array, or None if input is None.

        Raises:
            TypeError: If the input is not a numpy array.
            ValueError: If the mask batch shape is invalid.

        Example:
            >>> import numpy as np
            >>> validator = NumpyVideoBatchValidator()
            >>> masks = np.random.randint(0, 2, size=(2, 5, 224, 224))  # [N, T, H, W]
            >>> validated_masks = validator.validate_gt_mask(masks)
            >>> print(validated_masks.shape, validated_masks.dtype)
            (2, 5, 224, 224) bool
        """
        if gt_mask is None:
            return None
        if not isinstance(gt_mask, np.ndarray):
            msg = f"Ground truth mask batch must be a numpy.ndarray, got {type(gt_mask)}."
            raise TypeError(msg)
        if gt_mask.ndim not in {4, 5}:
            msg = f"Ground truth mask batch must have shape [N, T, H, W] or [N, T, H, W, 1], got shape {gt_mask.shape}."
            raise ValueError(msg)
        if gt_mask.ndim == 5 and gt_mask.shape[4] != 1:
            msg = f"Ground truth mask batch must have 1 channel, got {gt_mask.shape[4]}."
            raise ValueError(msg)
        return gt_mask.astype(bool)

    @staticmethod
    def validate_mask_path(mask_path: Sequence[str] | None) -> list[str] | None:
        """Validate the mask paths for a batch.

        Args:
            mask_path (``Sequence[str]`` | ``None``): Input mask paths to validate.

        Returns:
            ``list[str]`` | ``None``: Validated mask paths, or None if input is None.

        Example:
            >>> validator = NumpyVideoBatchValidator()
            >>> paths = ["/path/to/mask1.png", "/path/to/mask2.png"]
            >>> validated_paths = validator.validate_mask_path(paths)
            >>> print(validated_paths)
            ['/path/to/mask1.png', '/path/to/mask2.png']
        """
        return validate_batch_path(mask_path)

    @staticmethod
    def validate_anomaly_map(anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate the anomaly map batch.

        Args:
            anomaly_map (``np.ndarray`` | ``None``): Input anomaly map batch to validate.

        Returns:
            ``np.ndarray`` | ``None``: Validated anomaly map batch as float32 numpy array,
                or None if input is None.

        Raises:
            TypeError: If the input is not a numpy array.
            ValueError: If the anomaly map batch shape is invalid.

        Example:
            >>> import numpy as np
            >>> validator = NumpyVideoBatchValidator()
            >>> anomaly_maps = np.random.rand(2, 5, 224, 224)  # [N, T, H, W]
            >>> validated_maps = validator.validate_anomaly_map(anomaly_maps)
            >>> print(validated_maps.shape, validated_maps.dtype)
            (2, 5, 224, 224) float32
        """
        if anomaly_map is None:
            return None
        if not isinstance(anomaly_map, np.ndarray):
            msg = f"Anomaly map batch must be a numpy.ndarray, got {type(anomaly_map)}."
            raise TypeError(msg)
        if anomaly_map.ndim not in {4, 5}:
            msg = f"Anomaly map batch must have shape [N, T, H, W] or [N, T, H, W, 1], got shape {anomaly_map.shape}."
            raise ValueError(msg)
        if anomaly_map.ndim == 5 and anomaly_map.shape[4] != 1:
            msg = f"Anomaly map batch must have 1 channel, got {anomaly_map.shape[4]}."
            raise ValueError(msg)
        return anomaly_map.astype(np.float32)

    @staticmethod
    def validate_pred_score(pred_score: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction scores for a batch.

        Args:
            pred_score (``np.ndarray`` | ``None``): Input prediction scores to validate.

        Returns:
            ``np.ndarray`` | ``None``: Validated prediction scores as float32 numpy array,
                or None if input is None.

        Raises:
            TypeError: If the input is not a numpy array.
            ValueError: If the prediction score batch shape is invalid.

        Example:
            >>> import numpy as np
            >>> validator = NumpyVideoBatchValidator()
            >>> scores = np.array([0.1, 0.8, 0.3, 0.6])
            >>> validated_scores = validator.validate_pred_score(scores)
            >>> print(validated_scores, validated_scores.dtype)
            [0.1 0.8 0.3 0.6] float32
        """
        if pred_score is None:
            return None
        if not isinstance(pred_score, np.ndarray):
            msg = f"Prediction score batch must be a numpy.ndarray, got {type(pred_score)}."
            raise TypeError(msg)
        if pred_score.ndim != 1:
            msg = f"Prediction score batch must be 1-dimensional, got shape {pred_score.shape}."
            raise ValueError(msg)
        return pred_score.astype(np.float32)

    @staticmethod
    def validate_pred_mask(pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction mask batch.

        Args:
            pred_mask (``np.ndarray`` | ``None``): Input prediction mask batch to validate.

        Returns:
            ``np.ndarray`` | ``None``: Validated prediction mask batch as boolean numpy
                array, or None if input is None.

        Example:
            >>> import numpy as np
            >>> validator = NumpyVideoBatchValidator()
            >>> masks = np.random.randint(0, 2, size=(2, 5, 224, 224))  # [N, T, H, W]
            >>> validated_masks = validator.validate_pred_mask(masks)
            >>> print(validated_masks.shape, validated_masks.dtype)
            (2, 5, 224, 224) bool
        """
        return NumpyVideoBatchValidator.validate_gt_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction label batch.

        Args:
            pred_label (``np.ndarray`` | ``None``): Input prediction label batch to
                validate.

        Returns:
            ``np.ndarray`` | ``None``: Validated prediction label batch as boolean numpy
                array, or None if input is None.

        Raises:
            TypeError: If the input is not a numpy array.
            ValueError: If the prediction label batch shape is invalid.

        Example:
            >>> import numpy as np
            >>> validator = NumpyVideoBatchValidator()
            >>> labels = np.array([0, 1, 1, 0])
            >>> validated_labels = validator.validate_pred_label(labels)
            >>> print(validated_labels, validated_labels.dtype)
            [False  True  True False] bool
        """
        if pred_label is None:
            return None
        if not isinstance(pred_label, np.ndarray):
            msg = f"Prediction label batch must be a numpy.ndarray, got {type(pred_label)}."
            raise TypeError(msg)
        if pred_label.ndim != 1:
            msg = f"Prediction label batch must be 1-dimensional, got shape {pred_label.shape}."
            raise ValueError(msg)
        return pred_label.astype(bool)

    @staticmethod
    def validate_video_path(video_path: list[str] | None) -> list[str] | None:
        """Validate the video paths for a batch.

        Args:
            video_path (``list[str]`` | ``None``): Input video paths to validate.

        Returns:
            ``list[str]`` | ``None``: Validated video paths, or None if input is None.

        Example:
            >>> validator = NumpyVideoBatchValidator()
            >>> paths = ["/path/to/video1.mp4", "/path/to/video2.mp4"]
            >>> validated_paths = validator.validate_video_path(paths)
            >>> print(validated_paths)
            ['/path/to/video1.mp4', '/path/to/video2.mp4']
        """
        return validate_batch_path(video_path)

    @staticmethod
    def validate_original_image(original_image: np.ndarray | None) -> np.ndarray | None:
        """Validate the original video batch.

        Args:
            original_image (``np.ndarray`` | ``None``): Input original video batch to
                validate.

        Returns:
            ``np.ndarray`` | ``None``: Validated original video batch, or None if input is
                None.

        Raises:
            TypeError: If the input is not a numpy array.
            ValueError: If the original image batch shape is invalid.

        Example:
            >>> import numpy as np
            >>> validator = NumpyVideoBatchValidator()
            >>> original_batch = np.random.rand(2, 10, 224, 224, 3)  # [N, T, H, W, C]
            >>> validated_batch = validator.validate_original_image(original_batch)
            >>> print(validated_batch.shape, validated_batch.dtype)
            (2, 10, 224, 224, 3) float64
        """
        if original_image is None:
            return None
        if not isinstance(original_image, np.ndarray):
            msg = f"Original image batch must be a numpy.ndarray, got {type(original_image)}."
            raise TypeError(msg)
        if original_image.ndim not in {4, 5}:
            msg = (
                "Original image batch must have shape [N, T, H, W, C] or [N, H, W, C], "
                f"got shape {original_image.shape}."
            )
            raise ValueError(msg)
        if original_image.shape[-1] != 3:
            msg = f"Original image batch must have 3 channels, got {original_image.shape[-1]}."
            raise ValueError(msg)
        return original_image

    @staticmethod
    def validate_target_frame(target_frame: np.ndarray | None) -> np.ndarray | None:
        """Validate the target frame indices for a batch.

        Args:
            target_frame (``np.ndarray`` | ``None``): Input target frame indices to
                validate.

        Returns:
            ``np.ndarray`` | ``None``: Validated target frame indices, or None if input is
                None.

        Raises:
            TypeError: If the input is not a numpy array of integers.
            ValueError: If the target frame indices are negative or the shape is invalid.

        Example:
            >>> import numpy as np
            >>> validator = NumpyVideoBatchValidator()
            >>> frames = np.array([0, 5, 2, 7])
            >>> validated_frames = validator.validate_target_frame(frames)
            >>> print(validated_frames, validated_frames.dtype)
            [0 5 2 7] int64
        """
        if target_frame is None:
            return None
        if not isinstance(target_frame, np.ndarray):
            msg = f"Target frame batch must be a numpy.ndarray, got {type(target_frame)}."
            raise TypeError(msg)
        if target_frame.ndim != 1:
            msg = f"Target frame batch must be 1-dimensional, got shape {target_frame.shape}."
            raise ValueError(msg)
        if not np.issubdtype(target_frame.dtype, np.integer):
            msg = f"Target frame batch must be integer type, got {target_frame.dtype}."
            raise TypeError(msg)
        if np.any(target_frame < 0):
            msg = "Target frame indices must be non-negative."
            raise ValueError(msg)
        return target_frame

    @staticmethod
    def validate_explanation(explanation: list[str] | None) -> list[str] | None:
        """Validate the explanation string."""
        return NumpyImageBatchValidator.validate_explanation(explanation)
