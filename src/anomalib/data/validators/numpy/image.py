"""Validate numpy image data.

This module provides validators for image data stored as numpy arrays. The validators
ensure data consistency and correctness for images and batches of images.

The validators check:
    - Array shapes and dimensions
    - Data types
    - Value ranges
    - Label formats
    - Mask properties

Example:
    Validate a single image::

        >>> from anomalib.data.validators import NumpyImageValidator
        >>> validator = NumpyImageValidator()
        >>> validator.validate_image(image)

    Validate a batch of images::

        >>> from anomalib.data.validators import NumpyImageBatchValidator
        >>> validator = NumpyImageBatchValidator()
        >>> validator(images=images, labels=labels, masks=masks)

Note:
    The validators are used internally by the data modules to ensure data
    consistency before processing image data.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np
from anomalib.data.validators.path import validate_path


class NumpyImageValidator:
    """Validate numpy array data for images.

    This class provides validation methods for image data stored as numpy arrays.
    It ensures data consistency and correctness for images and associated metadata.

    The validator checks:
        - Array shapes and dimensions
        - Data types
        - Value ranges
        - Label formats
        - Mask properties
        - Path validity

    Example:
        Validate an image and associated metadata::

            >>> from anomalib.data.validators import NumpyImageValidator
            >>> validator = NumpyImageValidator()
            >>> image = np.random.rand(256, 256, 3)
            >>> validated_image = validator.validate_image(image)
            >>> label = 1
            >>> validated_label = validator.validate_gt_label(label)
            >>> mask = np.random.randint(0, 2, (256, 256))
            >>> validated_mask = validator.validate_gt_mask(mask)

    Note:
        The validator is used internally by the data modules to ensure data
        consistency before processing.
    """

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Validate the image array.

        Validates and normalizes input image arrays. Handles both RGB and grayscale
        images, and converts between channel-first and channel-last formats.

        Args:
            image (``np.ndarray``): Input image array to validate.

        Returns:
            ``np.ndarray``: Validated image array in channel-last format (H,W,C).

        Raises:
            TypeError: If ``image`` is not a numpy array.
            ValueError: If ``image`` dimensions or channels are invalid.

        Example:
            Validate RGB and grayscale images::

                >>> import numpy as np
                >>> from anomalib.data.validators import NumpyImageValidator
                >>> rgb_image = np.random.rand(256, 256, 3)
                >>> validated_rgb = NumpyImageValidator.validate_image(rgb_image)
                >>> validated_rgb.shape
                (256, 256, 3)
                >>> gray_image = np.random.rand(256, 256)
                >>> validated_gray = NumpyImageValidator.validate_image(gray_image)
                >>> validated_gray.shape
                (256, 256, 1)

        Note:
            - 2D arrays are treated as grayscale and expanded to 3D
            - Channel-first arrays (C,H,W) are converted to channel-last (H,W,C)
            - Output is always float32 type
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

        # Check if the image is in torch style (C, H, W) and rearrange if necessary
        if image.shape[0] in {1, 3} and image.shape[2] not in {1, 3}:
            image = np.transpose(image, (1, 2, 0))

        if image.shape[2] not in {1, 3}:
            msg = f"Image must have 1 or 3 channels, got {image.shape[2]}."
            raise ValueError(msg)

        return image.astype(np.float32)

    @staticmethod
    def validate_gt_label(label: int | np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth label.

        Validates and normalizes input labels to boolean numpy arrays.

        Args:
            label (``int`` | ``np.ndarray`` | ``None``): Input ground truth label.

        Returns:
            ``np.ndarray`` | ``None``: Validated label as boolean array, or None.

        Raises:
            TypeError: If ``label`` is not an integer or numpy array.
            ValueError: If ``label`` shape is not scalar.

        Example:
            Validate integer and array labels::

                >>> import numpy as np
                >>> from anomalib.data.validators import NumpyImageValidator
                >>> label_int = 1
                >>> validated_label = NumpyImageValidator.validate_gt_label(label_int)
                >>> validated_label
                array(True)
                >>> label_array = np.array(0)
                >>> validated_label = NumpyImageValidator.validate_gt_label(label_array)
                >>> validated_label
                array(False)

        Note:
            - Integer inputs are converted to numpy arrays
            - Output is always boolean type
            - None inputs return None
        """
        if label is None:
            return None
        if isinstance(label, int | np.bool_):
            label = np.array(label)
        if not isinstance(label, np.ndarray):
            msg = f"Ground truth label must be an integer or a numpy.ndarray, got {type(label)}."
            raise TypeError(msg)
        if label.ndim != 0:
            msg = f"Ground truth label must be a scalar, got shape {label.shape}."
            raise ValueError(msg)
        if not np.issubdtype(label.dtype, np.integer) and not np.issubdtype(label.dtype, bool):
            msg = f"Ground truth label must be boolean or integer, got {label.dtype}."
            raise TypeError(msg)
        return label.astype(bool)

    @staticmethod
    def validate_gt_mask(mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth mask.

        Validates and normalizes input mask arrays.

        Args:
            mask (``np.ndarray`` | ``None``): Input ground truth mask.

        Returns:
            ``np.ndarray`` | ``None``: Validated mask as boolean array, or None.

        Raises:
            TypeError: If ``mask`` is not a numpy array.
            ValueError: If ``mask`` dimensions are invalid.

        Example:
            Validate a binary mask::

                >>> import numpy as np
                >>> from anomalib.data.validators import NumpyImageValidator
                >>> mask = np.random.randint(0, 2, (224, 224))
                >>> validated_mask = NumpyImageValidator.validate_gt_mask(mask)
                >>> validated_mask.shape
                (224, 224)

        Note:
            - 3D masks with shape (H,W,1) are squeezed to (H,W)
            - Output is always boolean type
            - None inputs return None
        """
        if mask is None:
            return None
        if not isinstance(mask, np.ndarray):
            msg = f"Mask must be a numpy.ndarray, got {type(mask)}."
            raise TypeError(msg)
        if mask.ndim not in {2, 3}:
            msg = f"Mask must have shape [H, W] or [H, W, 1] got shape {mask.shape}."
            raise ValueError(msg)
        if mask.ndim == 3:
            if mask.shape[2] != 1:
                msg = f"Mask must have 1 channel, got {mask.shape[2]}."
                raise ValueError(msg)
            mask = mask.squeeze(2)
        return mask.astype(bool)

    @staticmethod
    def validate_anomaly_map(anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate the anomaly map.

        Validates and normalizes input anomaly map arrays.

        Args:
            anomaly_map (``np.ndarray`` | ``None``): Input anomaly map.

        Returns:
            ``np.ndarray`` | ``None``: Validated anomaly map as float32 array, or None.

        Raises:
            TypeError: If ``anomaly_map`` is not a numpy array.
            ValueError: If ``anomaly_map`` dimensions are invalid.

        Example:
            Validate an anomaly map::

                >>> import numpy as np
                >>> from anomalib.data.validators import NumpyImageValidator
                >>> anomaly_map = np.random.rand(224, 224)
                >>> validated_map = NumpyImageValidator.validate_anomaly_map(anomaly_map)
                >>> validated_map.shape
                (224, 224)

        Note:
            - 3D maps with shape (1,H,W) are squeezed to (H,W)
            - Output is always float32 type
            - None inputs return None
        """
        if anomaly_map is None:
            return None
        if not isinstance(anomaly_map, np.ndarray):
            msg = f"Anomaly map must be a numpy array, got {type(anomaly_map)}."
            raise TypeError(msg)
        if anomaly_map.ndim not in {2, 3}:
            msg = f"Anomaly map must have shape [H, W] or [1, H, W], got shape {anomaly_map.shape}."
            raise ValueError(msg)
        if anomaly_map.ndim == 3:
            if anomaly_map.shape[0] != 1:
                msg = f"Anomaly map with 3 dimensions must have 1 channel, got {anomaly_map.shape[0]}."
                raise ValueError(msg)
            anomaly_map = anomaly_map.squeeze(0)
        return anomaly_map.astype(np.float32)

    @staticmethod
    def validate_image_path(image_path: str | None) -> str | None:
        """Validate the image path.

        Args:
            image_path (``str`` | ``None``): Input image path.

        Returns:
            ``str`` | ``None``: Validated image path, or None.

        Example:
            Validate an image path::

                >>> from anomalib.data.validators import NumpyImageValidator
                >>> path = "/path/to/image.jpg"
                >>> validated_path = NumpyImageValidator.validate_image_path(path)
                >>> validated_path == path
                True

        Note:
            Returns None if input is None.
        """
        return validate_path(image_path) if image_path else None

    @staticmethod
    def validate_mask_path(mask_path: str | None) -> str | None:
        """Validate the mask path.

        Args:
            mask_path (``str`` | ``None``): Input mask path.

        Returns:
            ``str`` | ``None``: Validated mask path, or None.

        Example:
            Validate a mask path::

                >>> from anomalib.data.validators import NumpyImageValidator
                >>> path = "/path/to/mask.png"
                >>> validated_path = NumpyImageValidator.validate_mask_path(path)
                >>> validated_path == path
                True

        Note:
            Returns None if input is None.
        """
        return validate_path(mask_path) if mask_path else None

    @staticmethod
    def validate_pred_score(
        pred_score: np.ndarray | float | None,
        anomaly_map: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Validate the prediction score.

        Validates and normalizes prediction scores to float32 numpy arrays.

        Args:
            pred_score (``np.ndarray`` | ``float`` | ``None``): Input prediction score.
            anomaly_map (``np.ndarray`` | ``None``): Input anomaly map.

        Returns:
            ``np.ndarray`` | ``None``: Validated score as float32 array, or None.

        Raises:
            TypeError: If ``pred_score`` cannot be converted to numpy array.
            ValueError: If ``pred_score`` is not scalar.

        Example:
            Validate prediction scores::

                >>> import numpy as np
                >>> from anomalib.data.validators import NumpyImageValidator
                >>> score = 0.8
                >>> validated_score = NumpyImageValidator.validate_pred_score(score)
                >>> validated_score
                array(0.8, dtype=float32)
                >>> score_array = np.array(0.7)
                >>> validated_score = NumpyImageValidator.validate_pred_score(score_array)
                >>> validated_score
                array(0.7, dtype=float32)

        Note:
            - If input is None and anomaly_map provided, returns max of anomaly_map
            - Output is always float32 type
            - None inputs with no anomaly_map return None
        """
        if pred_score is None:
            return np.amax(anomaly_map) if anomaly_map is not None else None

        if not isinstance(pred_score, np.ndarray):
            try:
                pred_score = np.array(pred_score)
            except Exception as e:
                msg = "Failed to convert pred_score to a numpy.ndarray."
                raise ValueError(msg) from e
        pred_score = pred_score.squeeze()
        if pred_score.ndim != 0:
            msg = f"Predicted score must be a scalar, got shape {pred_score.shape}."
            raise ValueError(msg)

        return pred_score.astype(np.float32)

    @staticmethod
    def validate_pred_mask(pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction mask.

        Validates and normalizes prediction mask arrays.

        Args:
            pred_mask (``np.ndarray`` | ``None``): Input prediction mask.

        Returns:
            ``np.ndarray`` | ``None``: Validated mask as boolean array, or None.

        Example:
            Validate a prediction mask::

                >>> import numpy as np
                >>> from anomalib.data.validators import NumpyImageValidator
                >>> mask = np.random.randint(0, 2, (224, 224))
                >>> validated_mask = NumpyImageValidator.validate_pred_mask(mask)
                >>> validated_mask.shape
                (224, 224)

        Note:
            Uses same validation as ground truth masks.
        """
        return NumpyImageValidator.validate_gt_mask(pred_mask)  # We can reuse the gt_mask validation

    @staticmethod
    def validate_pred_label(pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction label.

        Validates and normalizes prediction labels to boolean numpy arrays.

        Args:
            pred_label (``np.ndarray`` | ``None``): Input prediction label.

        Returns:
            ``np.ndarray`` | ``None``: Validated label as boolean array, or None.

        Raises:
            TypeError: If ``pred_label`` cannot be converted to numpy array.
            ValueError: If ``pred_label`` is not scalar.

        Example:
            Validate a prediction label::

                >>> import numpy as np
                >>> from anomalib.data.validators import NumpyImageValidator
                >>> label = np.array(1)
                >>> validated_label = NumpyImageValidator.validate_pred_label(label)
                >>> validated_label
                array(True)

        Note:
            - Output is always boolean type
            - None inputs return None
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
    def validate_explanation(explanation: str | None) -> str | None:
        """Validate the explanation string.

        Args:
            explanation (``str`` | ``None``): Input explanation string.

        Returns:
            ``str`` | ``None``: Validated explanation string, or None.

        Raises:
            TypeError: If ``explanation`` is not a string.

        Example:
            Validate an explanation string::

                >>> from anomalib.dataclasses.validators import ImageValidator
                >>> explanation = "The image has a crack on the wall."
                >>> validated = ImageValidator.validate_explanation(explanation)
                >>> validated == explanation
                True

        Note:
            Returns None if input is None.
        """
        if explanation is None:
            return None
        if not isinstance(explanation, str):
            msg = f"Explanation must be a string, got {type(explanation)}."
            raise TypeError(msg)
        return explanation


class NumpyImageBatchValidator:
    """Validate batches of image data stored as numpy arrays.

    This class provides validation methods for batches of image data stored as numpy arrays.
    It ensures data consistency and correctness for images and associated metadata.

    The validator checks:
        - Array shapes and dimensions
        - Data types
        - Value ranges
        - Label formats
        - Mask properties
        - Path validity

    Example:
        Validate a batch of images and associated metadata::

            >>> from anomalib.data.validators import NumpyImageBatchValidator
            >>> validator = NumpyImageBatchValidator()
            >>> images = np.random.rand(32, 256, 256, 3)
            >>> labels = np.zeros(32)
            >>> masks = np.zeros((32, 256, 256))
            >>> validator.validate_image(images)
            >>> validator.validate_gt_label(labels)
            >>> validator.validate_gt_mask(masks)
    """

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Validate the image batch array.

        This method validates batches of images stored as numpy arrays. It handles:
            - Single images and batches
            - Grayscale and RGB images
            - Channel-first and channel-last formats
            - Type conversion to float32

        Args:
            image (``np.ndarray``): Input image batch array.

        Returns:
            ``np.ndarray``: Validated image batch array in [N,H,W,C] format.

        Raises:
            TypeError: If ``image`` is not a numpy array.
            ValueError: If ``image`` shape is invalid.

        Examples:
            Validate RGB batch::

                >>> batch = np.random.rand(32, 224, 224, 3)
                >>> validated = NumpyImageBatchValidator.validate_image(batch)
                >>> validated.shape
                (32, 224, 224, 3)

            Validate grayscale batch::

                >>> gray = np.random.rand(32, 224, 224)
                >>> validated = NumpyImageBatchValidator.validate_image(gray)
                >>> validated.shape
                (32, 224, 224, 1)

            Validate channel-first batch::

                >>> chf = np.random.rand(32, 3, 224, 224)
                >>> validated = NumpyImageBatchValidator.validate_image(chf)
                >>> validated.shape
                (32, 224, 224, 3)

            Validate single image::

                >>> img = np.zeros((224, 224, 3))
                >>> validated = NumpyImageBatchValidator.validate_image(img)
                >>> validated.shape
                (1, 224, 224, 3)
        """
        # Check if the image is a numpy array
        if not isinstance(image, np.ndarray):
            msg = f"Image batch must be a numpy.ndarray, got {type(image)}."
            raise TypeError(msg)

        # Handle single image input
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] in {1, 3}):
            image = image[np.newaxis, ...]

        # Check if the image has the correct number of dimensions
        if image.ndim not in {3, 4}:
            msg = f"Image batch must have shape [N, H, W] or [N, H, W, C], got shape {image.shape}."
            raise ValueError(msg)

        # Handle 3D grayscale images
        if image.ndim == 3:
            image = image[..., np.newaxis]

        # Handle torch style (N, C, H, W) and rearrange if necessary
        if image.shape[1] in {1, 3} and image.shape[3] not in {1, 3}:
            image = np.transpose(image, (0, 2, 3, 1))

        # Check if the image has the correct number of channels
        if image.shape[-1] not in {1, 3}:
            msg = f"Image batch must have 1 or 3 channels, got {image.shape[-1]}."
            raise ValueError(msg)

        return image.astype(np.float32)

    @staticmethod
    def validate_gt_label(gt_label: np.ndarray | Sequence[int] | None) -> np.ndarray | None:
        """Validate the ground truth label batch.

        This method validates batches of ground truth labels. It handles:
            - Numpy arrays and sequences of integers
            - Type conversion to boolean
            - Shape validation

        Args:
            gt_label (``np.ndarray`` | ``Sequence[int]`` | ``None``): Input ground truth label
                batch.

        Returns:
            ``np.ndarray`` | ``None``: Validated ground truth label batch as boolean array,
                or ``None``.

        Raises:
            TypeError: If ``gt_label`` is not a numpy array or sequence of integers.
            ValueError: If ``gt_label`` shape is invalid.

        Examples:
            Validate numpy array labels::

                >>> labels = np.array([0, 1, 1, 0])
                >>> validated = NumpyImageBatchValidator.validate_gt_label(labels)
                >>> validated
                array([False,  True,  True, False])

            Validate list labels::

                >>> labels = [1, 0, 1, 1]
                >>> validated = NumpyImageBatchValidator.validate_gt_label(labels)
                >>> validated
                array([ True, False,  True,  True])
        """
        if gt_label is None:
            return None
        if isinstance(gt_label, Sequence) and not isinstance(gt_label, np.ndarray):
            gt_label = np.array(gt_label)
        if not isinstance(gt_label, np.ndarray):
            msg = f"Ground truth label batch must be a numpy.ndarray or Sequence[int], got {type(gt_label)}."
            raise TypeError(msg)
        if gt_label.ndim != 1:
            msg = f"Ground truth label batch must be 1-dimensional, got shape {gt_label.shape}."
            raise ValueError(msg)
        return gt_label.astype(bool)

    @staticmethod
    def validate_gt_mask(gt_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth mask batch.

        This method validates batches of ground truth masks. It handles:
            - Channel-first and channel-last formats
            - Type conversion to boolean
            - Shape validation

        Args:
            gt_mask (``np.ndarray`` | ``None``): Input ground truth mask batch.

        Returns:
            ``np.ndarray`` | ``None``: Validated ground truth mask batch as boolean array,
                or ``None``.

        Raises:
            TypeError: If ``gt_mask`` is not a numpy array.
            ValueError: If ``gt_mask`` shape is invalid.

        Examples:
            Validate channel-last masks::

                >>> masks = np.random.randint(0, 2, (4, 224, 224))
                >>> validated = NumpyImageBatchValidator.validate_gt_mask(masks)
                >>> validated.shape
                (4, 224, 224)
                >>> validated.dtype
                dtype('bool')

            Validate channel-first masks::

                >>> masks = np.random.randint(0, 2, (4, 1, 224, 224))
                >>> validated = NumpyImageBatchValidator.validate_gt_mask(masks)
                >>> validated.shape
                (4, 224, 224, 1)
        """
        if gt_mask is None:
            return None
        if not isinstance(gt_mask, np.ndarray):
            msg = f"Ground truth mask batch must be a numpy.ndarray, got {type(gt_mask)}."
            raise TypeError(msg)
        if gt_mask.ndim not in {3, 4}:
            msg = f"Ground truth mask batch must have shape [N, H, W] or [N, H, W, 1], got shape {gt_mask.shape}."
            raise ValueError(msg)

        # Check if the mask is in [N, H, W, 1] format and rearrange if necessary
        if gt_mask.ndim == 4 and gt_mask.shape[3] != 1:
            gt_mask = np.transpose(gt_mask, (0, 2, 3, 1))

        if gt_mask.ndim == 4 and gt_mask.shape[3] != 1:
            msg = f"Ground truth mask batch must have 1 channel, got {gt_mask.shape[3]}."
            raise ValueError(msg)

        return gt_mask.astype(bool)

    @staticmethod
    def validate_mask_path(mask_path: Sequence[str] | None) -> list[str] | None:
        """Validate the mask paths for a batch.

        This method validates sequences of mask file paths. It handles:
            - Type conversion to strings
            - Path sequence validation

        Args:
            mask_path (``Sequence[str]`` | ``None``): Input sequence of mask paths.

        Returns:
            ``list[str]`` | ``None``: Validated list of mask paths, or ``None``.

        Raises:
            TypeError: If ``mask_path`` is not a sequence of strings.

        Examples:
            Validate list of paths::

                >>> paths = ['mask1.png', 'mask2.png', 'mask3.png']
                >>> validated = NumpyImageBatchValidator.validate_mask_path(paths)
                >>> validated
                ['mask1.png', 'mask2.png', 'mask3.png']
        """
        if mask_path is None:
            return None
        if not isinstance(mask_path, Sequence):
            msg = f"Mask path must be a sequence of paths or strings, got {type(mask_path)}."
            raise TypeError(msg)
        return [str(path) for path in mask_path]

    @staticmethod
    def validate_anomaly_map(anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate the anomaly map batch.

        This method validates batches of anomaly maps. It handles:
            - Channel-first and channel-last formats
            - Type conversion to float32
            - Shape validation

        Args:
            anomaly_map (``np.ndarray`` | ``None``): Input anomaly map batch.

        Returns:
            ``np.ndarray`` | ``None``: Validated anomaly map batch, or ``None``.

        Raises:
            TypeError: If ``anomaly_map`` is not a numpy array.
            ValueError: If ``anomaly_map`` shape is invalid.

        Examples:
            Validate channel-last maps::

                >>> maps = np.random.rand(4, 224, 224)
                >>> validated = NumpyImageBatchValidator.validate_anomaly_map(maps)
                >>> validated.shape
                (4, 224, 224)
                >>> validated.dtype
                dtype('float32')

            Validate channel-first maps::

                >>> maps = np.random.rand(4, 1, 224, 224)
                >>> validated = NumpyImageBatchValidator.validate_anomaly_map(maps)
                >>> validated.shape
                (4, 224, 224, 1)
        """
        if anomaly_map is None:
            return None
        if not isinstance(anomaly_map, np.ndarray):
            msg = f"Anomaly map batch must be a numpy.ndarray, got {type(anomaly_map)}."
            raise TypeError(msg)
        if anomaly_map.ndim not in {3, 4}:
            msg = f"Anomaly map batch must have shape [N, H, W] or [N, H, W, 1], got shape {anomaly_map.shape}."
            raise ValueError(msg)
        # Check if the anomaly map is in [N, C, H, W] format and rearrange if necessary
        if anomaly_map.ndim == 4 and anomaly_map.shape[1] not in {1, 3}:
            anomaly_map = np.transpose(anomaly_map, (0, 2, 3, 1))
        return anomaly_map.astype(np.float32)

    @staticmethod
    def validate_pred_score(pred_score: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction scores for a batch.

        This method validates batches of prediction scores. It handles:
            - 1D and 2D arrays
            - Type conversion to float32
            - Shape validation

        Args:
            pred_score (``np.ndarray`` | ``None``): Input prediction score batch.

        Returns:
            ``np.ndarray`` | ``None``: Validated prediction score batch, or ``None``.

        Raises:
            TypeError: If ``pred_score`` is not a numpy array.
            ValueError: If ``pred_score`` shape is invalid.

        Examples:
            Validate 1D scores::

                >>> scores = np.array([0.1, 0.8, 0.3, 0.6])
                >>> validated = NumpyImageBatchValidator.validate_pred_score(scores)
                >>> validated
                array([0.1, 0.8, 0.3, 0.6], dtype=float32)

            Validate 2D scores::

                >>> scores = np.array([[0.1], [0.8], [0.3], [0.6]])
                >>> validated = NumpyImageBatchValidator.validate_pred_score(scores)
                >>> validated
                array([[0.1],
                       [0.8],
                       [0.3],
                       [0.6]], dtype=float32)
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
        """Validate the prediction mask batch.

        This method validates batches of prediction masks. It handles:
            - Channel-first and channel-last formats
            - Type conversion to boolean
            - Shape validation

        Args:
            pred_mask (``np.ndarray`` | ``None``): Input prediction mask batch.

        Returns:
            ``np.ndarray`` | ``None``: Validated prediction mask batch, or ``None``.

        Raises:
            TypeError: If ``pred_mask`` is not a numpy array.
            ValueError: If ``pred_mask`` shape is invalid.

        Examples:
            Validate channel-last masks::

                >>> masks = np.random.randint(0, 2, (4, 224, 224))
                >>> validated = NumpyImageBatchValidator.validate_pred_mask(masks)
                >>> validated.shape
                (4, 224, 224)
                >>> validated.dtype
                dtype('bool')

            Validate channel-first masks::

                >>> masks = np.random.randint(0, 2, (4, 1, 224, 224))
                >>> validated = NumpyImageBatchValidator.validate_pred_mask(masks)
                >>> validated.shape
                (4, 224, 224, 1)
        """
        return NumpyImageBatchValidator.validate_gt_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction label batch.

        This method validates batches of prediction labels. It handles:
            - 1D and 2D arrays
            - Type conversion to boolean
            - Shape validation

        Args:
            pred_label (``np.ndarray`` | ``None``): Input prediction label batch.

        Returns:
            ``np.ndarray`` | ``None``: Validated prediction label batch as boolean array,
                or ``None``.

        Raises:
            TypeError: If ``pred_label`` is not a numpy array.
            ValueError: If ``pred_label`` shape is invalid.

        Examples:
            Validate 1D labels::

                >>> labels = np.array([0, 1, 1, 0])
                >>> validated = NumpyImageBatchValidator.validate_pred_label(labels)
                >>> validated
                array([False,  True,  True, False])

            Validate 2D labels::

                >>> labels = np.array([[0], [1], [1], [0]])
                >>> validated = NumpyImageBatchValidator.validate_pred_label(labels)
                >>> validated
                array([[False],
                       [ True],
                       [ True],
                       [False]])
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
        """Validate the image paths for a batch.

        This method validates lists of image file paths. It handles:
            - Type conversion to strings
            - Path list validation

        Args:
            image_path (``list[str]`` | ``None``): Input list of image paths.

        Returns:
            ``list[str]`` | ``None``: Validated list of image paths, or ``None``.

        Raises:
            TypeError: If ``image_path`` is not a list.

        Examples:
            Validate list of paths::

                >>> paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
                >>> validated = NumpyImageBatchValidator.validate_image_path(paths)
                >>> validated
                ['image1.jpg', 'image2.jpg', 'image3.jpg']

            Validate mixed type paths::

                >>> paths = ['image1.jpg', 2, 'image3.jpg']
                >>> validated = NumpyImageBatchValidator.validate_image_path(paths)
                >>> validated
                ['image1.jpg', '2', 'image3.jpg']
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

        This method validates lists of explanation strings. It handles:
            - Type conversion to strings
            - List validation

        Args:
            explanation (``list[str]`` | ``None``): Input list of explanations.

        Returns:
            ``list[str]`` | ``None``: Validated list of explanations, or ``None``.

        Raises:
            TypeError: If ``explanation`` is not a list.

        Examples:
            Validate list of explanations::

                >>> explanations = ["The image has a crack.", "The image has a dent."]
                >>> validated = NumpyImageBatchValidator.validate_explanation(explanations)
                >>> validated
                ['The image has a crack.', 'The image has a dent.']
        """
        if explanation is None:
            return None
        if not isinstance(explanation, list):
            msg = f"Explanation must be a list of strings, got {type(explanation)}."
            raise TypeError(msg)
        return [str(exp) for exp in explanation]
