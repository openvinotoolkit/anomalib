"""Validate numpy image data."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np
from anomalib.data.validators.path import validate_path


class NumpyImageValidator:
    """Validate numpy.ndarray data for images."""

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Validate the image array.

        Args:
            image (np.ndarray): Input image array.

        Returns:
            np.ndarray: Validated image array.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the image array does not have the correct shape.

        Examples:
            >>> import numpy as np
            >>> from anomalib.data.validators.numpy.image import NumpyImageValidator
            >>> rgb_image = np.random.rand(256, 256, 3)
            >>> validated_rgb = NumpyImageValidator.validate_image(rgb_image)
            >>> validated_rgb.shape
            (256, 256, 3)
            >>> gray_image = np.random.rand(256, 256)
            >>> validated_gray = NumpyImageValidator.validate_image(gray_image)
            >>> validated_gray.shape
            (256, 256, 1)
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

        Args:
            label (int | np.ndarray | None): Input ground truth label.

        Returns:
            np.ndarray | None: Validated ground truth label as a boolean array, or None.

        Raises:
            TypeError: If the input is neither an integer nor a numpy.ndarray.
            ValueError: If the label shape or dtype is invalid.

        Examples:
            >>> import numpy as np
            >>> from anomalib.data.validators.numpy.image import NumpyImageValidator
            >>> label_int = 1
            >>> validated_label = NumpyImageValidator.validate_gt_label(label_int)
            >>> validated_label
            array(True)
            >>> label_array = np.array(0)
            >>> validated_label = NumpyImageValidator.validate_gt_label(label_array)
            >>> validated_label
            array(False)
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

        Args:
            mask (np.ndarray | None): Input ground truth mask.

        Returns:
            np.ndarray | None: Validated ground truth mask, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the mask shape is invalid.

        Examples:
            >>> import numpy as np
            >>> from anomalib.data.validators.numpy.image import NumpyImageValidator
            >>> mask = np.random.randint(0, 2, (224, 224))
            >>> validated_mask = NumpyImageValidator.validate_gt_mask(mask)
            >>> validated_mask.shape
            (224, 224)
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

        Args:
            anomaly_map (np.ndarray | None): Input anomaly map.

        Returns:
            np.ndarray | None: Validated anomaly map, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the anomaly map shape is invalid.

        Examples:
            >>> import numpy as np
            >>> from anomalib.data.validators.numpy.image import NumpyImageValidator
            >>> anomaly_map = np.random.rand(224, 224)
            >>> validated_map = NumpyImageValidator.validate_anomaly_map(anomaly_map)
            >>> validated_map.shape
            (224, 224)
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
            image_path (str | None): Input image path.

        Returns:
            str | None: Validated image path, or None.

        Examples:
            >>> from anomalib.data.validators.numpy.image import NumpyImageValidator
            >>> path = "/path/to/image.jpg"
            >>> validated_path = NumpyImageValidator.validate_image_path(path)
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
            >>> from anomalib.data.validators.numpy.image import NumpyImageValidator
            >>> path = "/path/to/mask.png"
            >>> validated_path = NumpyImageValidator.validate_mask_path(path)
            >>> validated_path == path
            True
        """
        return validate_path(mask_path) if mask_path else None

    @staticmethod
    def validate_pred_score(
        pred_score: np.ndarray | float | None,
        anomaly_map: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Validate the prediction score.

        Args:
            pred_score (np.ndarray | float | None): Input prediction score.
            anomaly_map (np.ndarray | None): Input anomaly map.

        Returns:
            np.ndarray | None: Validated prediction score as a float32 array, or None.

        Raises:
            TypeError: If the input is neither a float, numpy.ndarray, nor None.
            ValueError: If the prediction score is not a scalar.

        Examples:
            >>> import numpy as np
            >>> from anomalib.data.validators.numpy.image import NumpyImageValidator
            >>> score = 0.8
            >>> validated_score = NumpyImageValidator.validate_pred_score(score)
            >>> validated_score
            array(0.8, dtype=float32)
            >>> score_array = np.array(0.7)
            >>> validated_score = NumpyImageValidator.validate_pred_score(score_array)
            >>> validated_score
            array(0.7, dtype=float32)
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

        Args:
            pred_mask (np.ndarray | None): Input prediction mask.

        Returns:
            np.ndarray | None: Validated prediction mask, or None.

        Examples:
            >>> import numpy as np
            >>> from anomalib.data.validators.numpy.image import NumpyImageValidator
            >>> mask = np.random.randint(0, 2, (224, 224))
            >>> validated_mask = NumpyImageValidator.validate_pred_mask(mask)
            >>> validated_mask.shape
            (224, 224)
        """
        return NumpyImageValidator.validate_gt_mask(pred_mask)  # We can reuse the gt_mask validation

    @staticmethod
    def validate_pred_label(pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction label.

        Args:
            pred_label (np.ndarray | None): Input prediction label.

        Returns:
            np.ndarray | None: Validated prediction label as a boolean array, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the prediction label is not a scalar.

        Examples:
            >>> import numpy as np
            >>> from anomalib.data.validators.numpy.image import NumpyImageValidator
            >>> label = np.array(1)
            >>> validated_label = NumpyImageValidator.validate_pred_label(label)
            >>> validated_label
            array(True)
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
        """Validate the explanation.

        Args:
            explanation (str | None): Input explanation.

        Returns:
            str | None: Validated explanation, or None.

        Examples:
            >>> from anomalib.dataclasses.validators import ImageValidator
            >>> explanation = "The image has a crack on the wall."
            >>> validated_explanation = ImageValidator.validate_explanation(explanation)
            >>> validated_explanation == explanation
            True
        """
        if explanation is None:
            return None
        if not isinstance(explanation, str):
            msg = f"Explanation must be a string, got {type(explanation)}."
            raise TypeError(msg)
        return explanation


class NumpyImageBatchValidator:
    """Validate numpy.ndarray data for batches of images."""

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Validate the image batch array.

        Args:
            image (np.ndarray): Input image batch array.

        Returns:
            np.ndarray: Validated image batch array.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the image batch array does not have the correct shape.

        Examples:
            >>> import numpy as np
            >>> from anomalib.data.validators.numpy.image import NumpyImageBatchValidator
            >>> batch = np.random.rand(32, 224, 224, 3)
            >>> validated_batch = NumpyImageBatchValidator.validate_image(batch)
            >>> validated_batch.shape
            (32, 224, 224, 3)
            >>> grayscale_batch = np.random.rand(32, 224, 224)
            >>> validated_grayscale = NumpyImageBatchValidator.validate_image(grayscale_batch)
            >>> validated_grayscale.shape
            (32, 224, 224, 1)
            >>> torch_style_batch = np.random.rand(32, 3, 224, 224)
            >>> validated_torch_style = NumpyImageBatchValidator.validate_image(torch_style_batch)
            >>> validated_torch_style.shape
            (32, 224, 224, 3)
            >>> single_image = np.zeros((224, 224, 3))
            >>> validated_single = NumpyImageBatchValidator.validate_image(single_image)
            >>> validated_single.shape
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

        Args:
            gt_label (np.ndarray | Sequence[int] | None): Input ground truth label batch.

        Returns:
            np.ndarray | None: Validated ground truth label batch as a boolean array, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray or Sequence[int].
            ValueError: If the label batch shape is invalid.

        Examples:
            >>> import numpy as np
            >>> from anomalib.data.validators.numpy.image import NumpyImageBatchValidator
            >>> labels = np.array([0, 1, 1, 0])
            >>> validated_labels = NumpyImageBatchValidator.validate_gt_label(labels)
            >>> validated_labels
            array([False,  True,  True, False])
            >>> list_labels = [1, 0, 1, 1]
            >>> validated_list = NumpyImageBatchValidator.validate_gt_label(list_labels)
            >>> validated_list
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

        Args:
            gt_mask (np.ndarray | None): Input ground truth mask batch.

        Returns:
            np.ndarray | None: Validated ground truth mask batch as a boolean array, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the mask batch shape is invalid.

        Examples:
            >>> import numpy as np
            >>> from anomalib.data.validators.numpy.image import NumpyImageBatchValidator
            >>> masks = np.random.randint(0, 2, (4, 224, 224))
            >>> validated_masks = NumpyImageBatchValidator.validate_gt_mask(masks)
            >>> validated_masks.shape
            (4, 224, 224)
            >>> validated_masks.dtype
            dtype('bool')
            >>> torch_style_masks = np.random.randint(0, 2, (4, 1, 224, 224))
            >>> validated_torch_style = NumpyImageBatchValidator.validate_gt_mask(torch_style_masks)
            >>> validated_torch_style.shape
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

        Args:
            mask_path (Sequence[str] | None): Input sequence of mask paths.

        Returns:
            list[str] | None: Validated list of mask paths, or None.

        Raises:
            TypeError: If the input is not a sequence of strings.
            ValueError: If the number of paths doesn't match the batch size.

        Examples:
            >>> from anomalib.data.validators.numpy.image import NumpyImageBatchValidator
            >>> paths = ['mask1.png', 'mask2.png', 'mask3.png', 'mask4.png']
            >>> validated_paths = NumpyImageBatchValidator.validate_mask_path(paths)
            >>> validated_paths
            ['mask1.png', 'mask2.png', 'mask3.png', 'mask4.png']
            >>> NumpyImageBatchValidator.validate_mask_path(['mask1.png', 'mask2.png'], 4)
            Traceback (most recent call last):
                ...
            ValueError: Invalid length for mask_path. Got length 2 for batch size 4.
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

        Args:
            anomaly_map (np.ndarray | None): Input anomaly map batch.

        Returns:
            np.ndarray | None: Validated anomaly map batch, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the anomaly map batch shape is invalid.

        Examples:
            >>> import numpy as np
            >>> from anomalib.data.validators.numpy.image import NumpyImageBatchValidator
            >>> anomaly_maps = np.random.rand(4, 224, 224)
            >>> validated_maps = NumpyImageBatchValidator.validate_anomaly_map(anomaly_maps)
            >>> validated_maps.shape
            (4, 224, 224)
            >>> validated_maps.dtype
            dtype('float32')
            >>> torch_style_maps = np.random.rand(4, 1, 224, 224)
            >>> validated_torch_style = NumpyImageBatchValidator.validate_anomaly_map(torch_style_maps)
            >>> validated_torch_style.shape
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

        Args:
            pred_score (np.ndarray | None): Input prediction score batch.

        Returns:
            np.ndarray | None: Validated prediction score batch, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the prediction score batch is not 1-dimensional or 2-dimensional.

        Examples:
            >>> import numpy as np
            >>> from anomalib.data.validators.numpy.image import NumpyImageBatchValidator
            >>> scores = np.array([0.1, 0.8, 0.3, 0.6])
            >>> validated_scores = NumpyImageBatchValidator.validate_pred_score(scores)
            >>> validated_scores
            array([0.1, 0.8, 0.3, 0.6], dtype=float32)
            >>> scores_2d = np.array([[0.1], [0.8], [0.3], [0.6]])
            >>> validated_scores_2d = NumpyImageBatchValidator.validate_pred_score(scores_2d)
            >>> validated_scores_2d
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

        Args:
            pred_mask (np.ndarray | None): Input prediction mask batch.

        Returns:
            np.ndarray | None: Validated prediction mask batch, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the prediction mask batch shape is invalid.

        Examples:
            >>> import numpy as np
            >>> from anomalib.data.validators.numpy.image import NumpyImageBatchValidator
            >>> masks = np.random.randint(0, 2, (4, 224, 224))
            >>> validated_masks = NumpyImageBatchValidator.validate_pred_mask(masks)
            >>> validated_masks.shape
            (4, 224, 224)
            >>> validated_masks.dtype
            dtype('bool')
            >>> torch_style_masks = np.random.randint(0, 2, (4, 1, 224, 224))
            >>> validated_torch_style = NumpyImageBatchValidator.validate_pred_mask(torch_style_masks)
            >>> validated_torch_style.shape
            (4, 224, 224, 1)
        """
        return NumpyImageBatchValidator.validate_gt_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction label batch.

        Args:
            pred_label (np.ndarray | None): Input prediction label batch.

        Returns:
            np.ndarray | None: Validated prediction label batch as a boolean array, or None.

        Raises:
            TypeError: If the input is not a numpy.ndarray.
            ValueError: If the prediction label batch is not 1-dimensional or 2-dimensional.

        Examples:
            >>> import numpy as np
            >>> from anomalib.data.validators.numpy.image import NumpyImageBatchValidator
            >>> labels = np.array([0, 1, 1, 0])
            >>> validated_labels = NumpyImageBatchValidator.validate_pred_label(labels)
            >>> validated_labels
            array([False,  True,  True, False])
            >>> labels_2d = np.array([[0], [1], [1], [0]])
            >>> validated_labels_2d = NumpyImageBatchValidator.validate_pred_label(labels_2d)
            >>> validated_labels_2d
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

        Args:
            image_path (list[str] | None): Input list of image paths.

        Returns:
            list[str] | None: Validated list of image paths, or None.

        Raises:
            TypeError: If the input is not a list of strings.

        Examples:
            >>> from anomalib.data.validators.numpy.image import NumpyImageBatchValidator
            >>> paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
            >>> validated_paths = NumpyImageBatchValidator.validate_image_path(paths)
            >>> validated_paths
            ['image1.jpg', 'image2.jpg', 'image3.jpg']
            >>> NumpyImageBatchValidator.validate_image_path(['image1.jpg', 2, 'image3.jpg'])
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

        Args:
            explanation (list[str] | None): Input list of explanations.

        Returns:
            list[str] | None: Validated list of explanations, or None.

        Raises:
            TypeError: If the input is not a list of strings.

        Examples:
            >>> from anomalib.data.validators.torch.image import ImageBatchValidator
            >>> explanations = ["The image has a crack on the wall.", "The image has a dent on the car."]
            >>> validated_explanations = ImageBatchValidator.validate_explanation(explanations)
            >>> print(validated_explanations)
            ['The image has a crack on the wall.', 'The image has a dent on the car.']
        """
        if explanation is None:
            return None
        if not isinstance(explanation, list):
            msg = f"Explanation must be a list of strings, got {type(explanation)}."
            raise TypeError(msg)
        return [str(exp) for exp in explanation]
