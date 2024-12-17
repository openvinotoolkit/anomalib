"""Validate numpy image data.

This module provides validators for image data stored as numpy arrays. It includes
two main validator classes:

- ``NumpyImageValidator``: Validates single image numpy arrays
- ``NumpyImageBatchValidator``: Validates batches of image numpy arrays

The validators check that inputs meet format requirements like shape, type, etc.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np
from anomalib.data.validators.path import validate_path


class NumpyImageValidator:
    """Validate numpy.ndarray data for images.

    This class provides methods to validate image data and associated metadata
    like labels, masks, etc. It ensures inputs meet format requirements.
    """

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Validate an image array.

        Args:
            image: Input image as numpy array

        Returns:
            Validated image array

        Raises:
            TypeError: If input is not a numpy.ndarray
            ValueError: If image array has incorrect shape

        Example:
            >>> import numpy as np
            >>> validator = NumpyImageValidator()
            >>> rgb_image = np.random.rand(32, 32, 3)
            >>> validated = validator.validate_image(rgb_image)
            >>> validated.shape
            (32, 32, 3)
        """
        if not isinstance(image, np.ndarray):
            msg = f"Image must be a numpy.ndarray, got {type(image)}."
            raise TypeError(msg)

        # Handle 2D grayscale images
        if image.ndim == 2:
            image = image[..., np.newaxis]

        if image.ndim != 3:
            msg = f"Image must have 2 or 3 dimensions, got shape {image.shape}."
            raise ValueError(msg)

        # Check if image is in torch style (C,H,W) and rearrange if necessary
        if image.shape[0] in {1, 3} and image.shape[2] not in {1, 3}:
            image = np.transpose(image, (1, 2, 0))

        if image.shape[2] not in {1, 3}:
            msg = f"Image must have 1 or 3 channels, got {image.shape[2]}."
            raise ValueError(msg)

        return image.astype(np.float32)

    @staticmethod
    def validate_gt_label(label: int | np.ndarray | None) -> np.ndarray | None:
        """Validate a ground truth label.

        Args:
            label: Input label as integer or numpy array

        Returns:
            Validated label array or None

        Raises:
            TypeError: If input is not an integer or numpy.ndarray
            ValueError: If label shape or dtype is invalid

        Example:
            >>> import numpy as np
            >>> validator = NumpyImageValidator()
            >>> label = 1
            >>> validated = validator.validate_gt_label(label)
            >>> validated
            array(True)
        """
        if label is None:
            return None
        if isinstance(label, int | np.bool_):
            label = np.array(label)
        if not isinstance(label, np.ndarray):
            msg = f"Label must be an integer or numpy.ndarray, got {type(label)}."
            raise TypeError(msg)
        if label.ndim != 0:
            msg = f"Label must be a scalar, got shape {label.shape}."
            raise ValueError(msg)
        if not np.issubdtype(label.dtype, np.integer) and not np.issubdtype(
            label.dtype,
            bool,
        ):
            msg = f"Label must be boolean or integer, got {label.dtype}."
            raise TypeError(msg)
        return label.astype(bool)

    @staticmethod
    def validate_gt_mask(mask: np.ndarray | None) -> np.ndarray | None:
        """Validate a ground truth mask.

        Args:
            mask: Input mask as numpy array

        Returns:
            Validated mask array or None

        Raises:
            TypeError: If input is not a numpy.ndarray
            ValueError: If mask shape is invalid

        Example:
            >>> import numpy as np
            >>> validator = NumpyImageValidator()
            >>> mask = np.random.randint(0, 2, (32, 32))
            >>> validated = validator.validate_gt_mask(mask)
            >>> validated.shape
            (32, 32)
        """
        if mask is None:
            return None
        if not isinstance(mask, np.ndarray):
            msg = f"Mask must be a numpy.ndarray, got {type(mask)}."
            raise TypeError(msg)
        if mask.ndim not in {2, 3}:
            msg = f"Mask must have shape [H,W] or [H,W,1], got {mask.shape}."
            raise ValueError(msg)
        if mask.ndim == 3:
            if mask.shape[2] != 1:
                msg = f"Mask must have 1 channel, got {mask.shape[2]}."
                raise ValueError(msg)
            mask = mask.squeeze(2)
        return mask.astype(bool)

    @staticmethod
    def validate_anomaly_map(anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate an anomaly map.

        Args:
            anomaly_map: Input anomaly map as numpy array

        Returns:
            Validated anomaly map array or None

        Raises:
            TypeError: If input is not a numpy.ndarray
            ValueError: If anomaly map shape is invalid

        Example:
            >>> import numpy as np
            >>> validator = NumpyImageValidator()
            >>> amap = np.random.rand(32, 32)
            >>> validated = validator.validate_anomaly_map(amap)
            >>> validated.shape
            (32, 32)
        """
        if anomaly_map is None:
            return None
        if not isinstance(anomaly_map, np.ndarray):
            msg = f"Anomaly map must be a numpy array, got {type(anomaly_map)}."
            raise TypeError(msg)
        if anomaly_map.ndim not in {2, 3}:
            msg = f"Anomaly map must have shape [H,W] or [1,H,W], got {anomaly_map.shape}."
            raise ValueError(msg)
        if anomaly_map.ndim == 3:
            if anomaly_map.shape[0] != 1:
                msg = f"Anomaly map must have 1 channel, got {anomaly_map.shape[0]}."
                raise ValueError(msg)
            anomaly_map = anomaly_map.squeeze(0)
        return anomaly_map.astype(np.float32)

    @staticmethod
    def validate_image_path(image_path: str | None) -> str | None:
        """Validate an image path.

        Args:
            image_path: Input image path

        Returns:
            Validated image path or None

        Example:
            >>> validator = NumpyImageValidator()
            >>> path = "/path/to/image.jpg"
            >>> validated = validator.validate_image_path(path)
            >>> validated == path
            True
        """
        return validate_path(image_path) if image_path else None

    @staticmethod
    def validate_mask_path(mask_path: str | None) -> str | None:
        """Validate a mask path.

        Args:
            mask_path: Input mask path

        Returns:
            Validated mask path or None

        Example:
            >>> validator = NumpyImageValidator()
            >>> path = "/path/to/mask.png"
            >>> validated = validator.validate_mask_path(path)
            >>> validated == path
            True
        """
        return validate_path(mask_path) if mask_path else None

    @staticmethod
    def validate_pred_score(
        pred_score: np.ndarray | float | None,
        anomaly_map: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Validate a prediction score.

        Args:
            pred_score: Input prediction score
            anomaly_map: Optional anomaly map to get score from

        Returns:
            Validated prediction score or None

        Raises:
            TypeError: If input is not a float or numpy.ndarray
            ValueError: If prediction score is not a scalar

        Example:
            >>> import numpy as np
            >>> validator = NumpyImageValidator()
            >>> score = 0.8
            >>> validated = validator.validate_pred_score(score)
            >>> validated
            array(0.8, dtype=float32)
        """
        if pred_score is None:
            return np.amax(anomaly_map) if anomaly_map is not None else None

        if not isinstance(pred_score, np.ndarray):
            try:
                pred_score = np.array(pred_score)
            except Exception as e:
                msg = "Failed to convert pred_score to numpy.ndarray."
                raise ValueError(msg) from e
        pred_score = pred_score.squeeze()
        if pred_score.ndim != 0:
            msg = f"Predicted score must be a scalar, got shape {pred_score.shape}."
            raise ValueError(msg)

        return pred_score.astype(np.float32)

    @staticmethod
    def validate_pred_mask(pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate a prediction mask.

        Args:
            pred_mask: Input prediction mask

        Returns:
            Validated prediction mask or None

        Example:
            >>> import numpy as np
            >>> validator = NumpyImageValidator()
            >>> mask = np.random.randint(0, 2, (32, 32))
            >>> validated = validator.validate_pred_mask(mask)
            >>> validated.shape
            (32, 32)
        """
        return NumpyImageValidator.validate_gt_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate a prediction label.

        Args:
            pred_label: Input prediction label

        Returns:
            Validated prediction label or None

        Raises:
            TypeError: If input is not a numpy.ndarray
            ValueError: If prediction label is not a scalar

        Example:
            >>> import numpy as np
            >>> validator = NumpyImageValidator()
            >>> label = np.array(1)
            >>> validated = validator.validate_pred_label(label)
            >>> validated
            array(True)
        """
        if pred_label is None:
            return None
        if not isinstance(pred_label, np.ndarray):
            try:
                pred_label = np.array(pred_label)
            except Exception as e:
                msg = "Failed to convert pred_label to numpy.ndarray."
                raise ValueError(msg) from e
        pred_label = pred_label.squeeze()
        if pred_label.ndim != 0:
            msg = f"Predicted label must be a scalar, got shape {pred_label.shape}."
            raise ValueError(msg)
        return pred_label.astype(bool)

    @staticmethod
    def validate_explanation(explanation: str | None) -> str | None:
        """Validate an explanation string.

        Args:
            explanation: Input explanation text

        Returns:
            Validated explanation or None

        Raises:
            TypeError: If input is not a string

        Example:
            >>> validator = NumpyImageValidator()
            >>> text = "Image shows a defect"
            >>> validated = validator.validate_explanation(text)
            >>> validated == text
            True
        """
        if explanation is None:
            return None
        if not isinstance(explanation, str):
            msg = f"Explanation must be a string, got {type(explanation)}."
            raise TypeError(msg)
        return explanation


class NumpyImageBatchValidator:
    """Validate numpy.ndarray data for batches of images.

    This class provides methods to validate batches of image data and associated
    metadata like labels, masks, etc. It ensures inputs meet format requirements.
    """

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Validate an image batch array.

        Args:
            image: Input image batch as numpy array

        Returns:
            Validated image batch array

        Raises:
            TypeError: If input is not a numpy.ndarray
            ValueError: If image batch has incorrect shape

        Example:
            >>> import numpy as np
            >>> validator = NumpyImageBatchValidator()
            >>> batch = np.random.rand(4, 32, 32, 3)
            >>> validated = validator.validate_image(batch)
            >>> validated.shape
            (4, 32, 32, 3)
        """
        if not isinstance(image, np.ndarray):
            msg = f"Image batch must be a numpy.ndarray, got {type(image)}."
            raise TypeError(msg)

        # Handle single image input
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] in {1, 3}):
            image = image[np.newaxis, ...]

        if image.ndim not in {3, 4}:
            msg = f"Image batch must have shape [N,H,W] or [N,H,W,C], got {image.shape}."
            raise ValueError(msg)

        # Handle 3D grayscale images
        if image.ndim == 3:
            image = image[..., np.newaxis]

        # Handle torch style (N,C,H,W) and rearrange if necessary
        if image.shape[1] in {1, 3} and image.shape[3] not in {1, 3}:
            image = np.transpose(image, (0, 2, 3, 1))

        if image.shape[-1] not in {1, 3}:
            msg = f"Image batch must have 1 or 3 channels, got {image.shape[-1]}."
            raise ValueError(msg)

        return image.astype(np.float32)

    @staticmethod
    def validate_gt_label(
        gt_label: np.ndarray | Sequence[int] | None,
    ) -> np.ndarray | None:
        """Validate a ground truth label batch.

        Args:
            gt_label: Input label batch

        Returns:
            Validated label batch array or None

        Raises:
            TypeError: If input is not a numpy.ndarray or Sequence[int]
            ValueError: If label batch shape is invalid

        Example:
            >>> import numpy as np
            >>> validator = NumpyImageBatchValidator()
            >>> labels = np.array([0, 1, 1, 0])
            >>> validated = validator.validate_gt_label(labels)
            >>> validated
            array([False,  True,  True, False])
        """
        if gt_label is None:
            return None
        if isinstance(gt_label, Sequence) and not isinstance(gt_label, np.ndarray):
            gt_label = np.array(gt_label)
        if not isinstance(gt_label, np.ndarray):
            msg = f"Label batch must be numpy.ndarray or Sequence[int], got {type(gt_label)}."
            raise TypeError(msg)
        if gt_label.ndim != 1:
            msg = f"Label batch must be 1-dimensional, got shape {gt_label.shape}."
            raise ValueError(msg)
        return gt_label.astype(bool)

    @staticmethod
    def validate_gt_mask(gt_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate a ground truth mask batch.

        Args:
            gt_mask: Input mask batch as numpy array

        Returns:
            Validated mask batch array or None

        Raises:
            TypeError: If input is not a numpy.ndarray
            ValueError: If mask batch shape is invalid

        Example:
            >>> import numpy as np
            >>> validator = NumpyImageBatchValidator()
            >>> masks = np.random.randint(0, 2, (4, 32, 32))
            >>> validated = validator.validate_gt_mask(masks)
            >>> validated.shape
            (4, 32, 32)
        """
        if gt_mask is None:
            return None
        if not isinstance(gt_mask, np.ndarray):
            msg = f"Mask batch must be a numpy.ndarray, got {type(gt_mask)}."
            raise TypeError(msg)
        if gt_mask.ndim not in {3, 4}:
            msg = f"Mask batch must have shape [N,H,W] or [N,H,W,1], got {gt_mask.shape}."
            raise ValueError(msg)

        # Check if mask is in [N,H,W,1] format and rearrange if necessary
        if gt_mask.ndim == 4 and gt_mask.shape[3] != 1:
            gt_mask = np.transpose(gt_mask, (0, 2, 3, 1))

        if gt_mask.ndim == 4 and gt_mask.shape[3] != 1:
            msg = f"Mask batch must have 1 channel, got {gt_mask.shape[3]}."
            raise ValueError(msg)

        return gt_mask.astype(bool)

    @staticmethod
    def validate_mask_path(mask_path: Sequence[str] | None) -> list[str] | None:
        """Validate mask paths for a batch.

        Args:
            mask_path: Input sequence of mask paths

        Returns:
            Validated list of mask paths or None

        Raises:
            TypeError: If input is not a sequence of strings

        Example:
            >>> validator = NumpyImageBatchValidator()
            >>> paths = ['mask1.png', 'mask2.png']
            >>> validated = validator.validate_mask_path(paths)
            >>> validated
            ['mask1.png', 'mask2.png']
        """
        if mask_path is None:
            return None
        if not isinstance(mask_path, Sequence):
            msg = f"Mask path must be a sequence of paths, got {type(mask_path)}."
            raise TypeError(msg)
        return [str(path) for path in mask_path]

    @staticmethod
    def validate_anomaly_map(anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate an anomaly map batch.

        Args:
            anomaly_map: Input anomaly map batch

        Returns:
            Validated anomaly map batch or None

        Raises:
            TypeError: If input is not a numpy.ndarray
            ValueError: If anomaly map batch shape is invalid

        Example:
            >>> import numpy as np
            >>> validator = NumpyImageBatchValidator()
            >>> maps = np.random.rand(4, 32, 32)
            >>> validated = validator.validate_anomaly_map(maps)
            >>> validated.shape
            (4, 32, 32)
        """
        if anomaly_map is None:
            return None
        if not isinstance(anomaly_map, np.ndarray):
            msg = f"Anomaly map batch must be a numpy.ndarray, got {type(anomaly_map)}."
            raise TypeError(msg)
        if anomaly_map.ndim not in {3, 4}:
            msg = f"Anomaly map batch must have shape [N,H,W] or [N,H,W,1], got {anomaly_map.shape}."
            raise ValueError(msg)
        # Check if anomaly map is in [N,C,H,W] format and rearrange if necessary
        if anomaly_map.ndim == 4 and anomaly_map.shape[1] not in {1, 3}:
            anomaly_map = np.transpose(anomaly_map, (0, 2, 3, 1))
        return anomaly_map.astype(np.float32)

    @staticmethod
    def validate_pred_score(pred_score: np.ndarray | None) -> np.ndarray | None:
        """Validate prediction scores for a batch.

        Args:
            pred_score: Input prediction score batch

        Returns:
            Validated prediction score batch or None

        Raises:
            TypeError: If input is not a numpy.ndarray
            ValueError: If prediction score batch shape is invalid

        Example:
            >>> import numpy as np
            >>> validator = NumpyImageBatchValidator()
            >>> scores = np.array([0.1, 0.8, 0.3])
            >>> validated = validator.validate_pred_score(scores)
            >>> validated
            array([0.1, 0.8, 0.3], dtype=float32)
        """
        if pred_score is None:
            return None
        if not isinstance(pred_score, np.ndarray):
            msg = f"Prediction score batch must be a numpy.ndarray, got {type(pred_score)}."
            raise TypeError(msg)
        if pred_score.ndim not in {1, 2}:
            msg = f"Prediction score batch must be 1D or 2D, got shape {pred_score.shape}."
            raise ValueError(msg)

        return pred_score.astype(np.float32)

    @staticmethod
    def validate_pred_mask(pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate a prediction mask batch.

        Args:
            pred_mask: Input prediction mask batch

        Returns:
            Validated prediction mask batch or None

        Example:
            >>> import numpy as np
            >>> validator = NumpyImageBatchValidator()
            >>> masks = np.random.randint(0, 2, (4, 32, 32))
            >>> validated = validator.validate_pred_mask(masks)
            >>> validated.shape
            (4, 32, 32)
        """
        return NumpyImageBatchValidator.validate_gt_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate a prediction label batch.

        Args:
            pred_label: Input prediction label batch

        Returns:
            Validated prediction label batch or None

        Raises:
            TypeError: If input is not a numpy.ndarray
            ValueError: If prediction label batch shape is invalid

        Example:
            >>> import numpy as np
            >>> validator = NumpyImageBatchValidator()
            >>> labels = np.array([0, 1, 1, 0])
            >>> validated = validator.validate_pred_label(labels)
            >>> validated
            array([False,  True,  True, False])
        """
        if pred_label is None:
            return None
        if not isinstance(pred_label, np.ndarray):
            msg = f"Prediction label batch must be a numpy.ndarray, got {type(pred_label)}."
            raise TypeError(msg)
        if pred_label.ndim not in {1, 2}:
            msg = f"Prediction label batch must be 1D or 2D, got shape {pred_label.shape}."
            raise ValueError(msg)
        return pred_label.astype(bool)

    @staticmethod
    def validate_image_path(image_path: list[str] | None) -> list[str] | None:
        """Validate image paths for a batch.

        Args:
            image_path: Input list of image paths

        Returns:
            Validated list of image paths or None

        Raises:
            TypeError: If input is not a list of strings

        Example:
            >>> validator = NumpyImageBatchValidator()
            >>> paths = ['image1.jpg', 'image2.jpg']
            >>> validated = validator.validate_image_path(paths)
            >>> validated
            ['image1.jpg', 'image2.jpg']
        """
        if image_path is None:
            return None
        if not isinstance(image_path, list):
            msg = f"Image path must be a list of strings, got {type(image_path)}."
            raise TypeError(msg)
        return [str(path) for path in image_path]

    @staticmethod
    def validate_explanation(explanation: list[str] | None) -> list[str] | None:
        """Validate explanations for a batch.

        Args:
            explanation: Input list of explanations

        Returns:
            Validated list of explanations or None

        Raises:
            TypeError: If input is not a list of strings

        Example:
            >>> validator = NumpyImageBatchValidator()
            >>> explanations = ["Defect found", "No defect"]
            >>> validated = validator.validate_explanation(explanations)
            >>> validated
            ['Defect found', 'No defect']
        """
        if explanation is None:
            return None
        if not isinstance(explanation, list):
            msg = f"Explanation must be a list of strings, got {type(explanation)}."
            raise TypeError(msg)
        return [str(exp) for exp in explanation]
