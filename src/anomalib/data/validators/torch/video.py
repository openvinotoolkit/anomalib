"""Validate PyTorch tensor data for videos.

This module provides validators for video data stored as PyTorch tensors. The validators
ensure data consistency and correctness for videos and their batches.

The validators check:
    - Tensor shapes and dimensions
    - Data types
    - Value ranges
    - Label formats
    - Mask properties
    - Path validity

Example:
    Validate a single video::

        >>> from anomalib.data.validators import VideoValidator
        >>> validator = VideoValidator()
        >>> validator.validate_image(video)

    Validate a batch of videos::

        >>> from anomalib.data.validators import VideoBatchValidator
        >>> validator = VideoBatchValidator()
        >>> validator(videos=videos, labels=labels, masks=masks)

Note:
    The validators are used internally by the data modules to ensure data
    consistency before processing video data.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from torchvision.transforms.v2.functional import to_dtype_image
from torchvision.tv_tensors import Mask, Video

import torch
from anomalib.data.validators.path import validate_batch_path, validate_path
from anomalib.data.validators.torch.image import ImageBatchValidator, ImageValidator


class VideoValidator:
    """Validate torch.Tensor data for videos.

    This class provides static methods to validate video data and related metadata stored as
    PyTorch tensors. The validators ensure data consistency and correctness by checking
    tensor shapes, dimensions, data types, and value ranges.

    The validator methods handle:
        - Video tensors
        - Ground truth labels and masks
        - Prediction scores, labels and masks
        - Video paths and metadata
        - Frame indices and timing information

    Each validation method performs thorough checks and returns properly formatted data
    ready for use in video processing pipelines.

    Example:
        >>> import torch
        >>> from anomalib.data.validators import VideoValidator
        >>> video = torch.rand(10, 3, 256, 256)  # 10 frames, RGB
        >>> validator = VideoValidator()
        >>> validated_video = validator.validate_image(video)
        >>> validated_video.shape
        torch.Size([10, 3, 256, 256])
    """

    @staticmethod
    def validate_image(image: torch.Tensor) -> torch.Tensor:
        """Validate a video tensor.

        Validates and normalizes video tensors, handling both single and multi-frame cases.
        Checks tensor type, dimensions, and channel count.

        Args:
            image (torch.Tensor): Input video tensor with shape either:
                - ``[C, H, W]`` for single frame
                - ``[T, C, H, W]`` for multiple frames
                where ``C`` is channels (1 or 3), ``H`` height, ``W`` width,
                and ``T`` number of frames.

        Returns:
            torch.Tensor: Validated and normalized video tensor.

        Raises:
            TypeError: If ``image`` is not a ``torch.Tensor``.
            ValueError: If tensor dimensions or channel count are invalid.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoValidator
            >>> # Multi-frame RGB video
            >>> video = torch.rand(10, 3, 256, 256)
            >>> validated = VideoValidator.validate_image(video)
            >>> validated.shape
            torch.Size([10, 3, 256, 256])

            >>> # Single RGB frame
            >>> frame = torch.rand(3, 256, 256)
            >>> validated = VideoValidator.validate_image(frame)
            >>> validated.shape
            torch.Size([1, 3, 256, 256])

            >>> # Single grayscale frame
            >>> gray = torch.rand(1, 256, 256)
            >>> validated = VideoValidator.validate_image(gray)
            >>> validated.shape
            torch.Size([1, 1, 256, 256])
        """
        if not isinstance(image, torch.Tensor):
            msg = f"Video must be a torch.Tensor, got {type(image)}."
            raise TypeError(msg)

        if image.dim() == 3:  # Single frame case (C, H, W)
            if image.shape[0] not in {1, 3}:
                msg = f"Video must have 1 or 3 channels for single frame, got {image.shape[0]}."
                raise ValueError(msg)
        elif image.dim() == 4:  # Multiple frames case (T, C, H, W)
            if image.shape[1] not in {1, 3}:
                msg = f"Video must have 1 or 3 channels, got {image.shape[1]}."
                raise ValueError(msg)
        else:
            msg = f"Video must have 3 or 4 dimensions, got {image.dim()}."
            raise ValueError(msg)

        return to_dtype_image(image, torch.float32, scale=True)

    @staticmethod
    def validate_gt_label(label: int | torch.Tensor | None) -> torch.Tensor | None:
        """Validate ground truth label.

        Validates and converts ground truth labels to boolean tensors.

        Args:
            label (int | torch.Tensor | None): Input label as either:
                - Integer (0 or 1)
                - Boolean tensor
                - Integer tensor
                - ``None``

        Returns:
            torch.Tensor | None: Validated boolean tensor label or ``None``.

        Raises:
            TypeError: If ``label`` is not an integer, tensor or ``None``.
            ValueError: If label shape or dtype is invalid.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoValidator
            >>> # Integer label
            >>> validated = VideoValidator.validate_gt_label(1)
            >>> validated
            tensor(True)

            >>> # Tensor label
            >>> label = torch.tensor([0, 0], dtype=torch.int32)
            >>> validated = VideoValidator.validate_gt_label(label)
            >>> validated
            tensor([False, False])
        """
        if label is None:
            return None
        if isinstance(label, int):
            label = torch.tensor(label)
        if not isinstance(label, torch.Tensor):
            msg = f"Ground truth label must be an integer or a torch.Tensor, got {type(label)}."
            raise TypeError(msg)
        if torch.is_floating_point(label):
            msg = f"Ground truth label must be boolean or integer, got {label.dtype}."
            raise TypeError(msg)
        return label.bool()

    @staticmethod
    def validate_gt_mask(mask: torch.Tensor | None) -> Mask | None:
        """Validate ground truth mask.

        Validates and converts ground truth masks to boolean Mask objects.

        Args:
            mask (torch.Tensor | None): Input mask tensor with shape either:
                - ``[H, W]`` for single frame
                - ``[T, H, W]`` for multiple frames
                - ``[T, 1, H, W]`` for multiple frames with channel dimension
                where ``H`` is height, ``W`` width, and ``T`` number of frames.

        Returns:
            Mask | None: Validated boolean mask or ``None``.

        Raises:
            TypeError: If ``mask`` is not a ``torch.Tensor`` or ``None``.
            ValueError: If mask shape is invalid.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoValidator
            >>> # Multi-frame mask
            >>> mask = torch.randint(0, 2, (10, 1, 224, 224))
            >>> validated = VideoValidator.validate_gt_mask(mask)
            >>> isinstance(validated, Mask)
            True
            >>> validated.shape
            torch.Size([10, 224, 224])
        """
        if mask is None:
            return None
        if not isinstance(mask, torch.Tensor):
            msg = f"Mask must be a torch.Tensor, got {type(mask)}."
            raise TypeError(msg)
        if mask.ndim not in {2, 3, 4}:
            msg = f"Mask must have shape [H, W], [T, H, W] or [T, 1, H, W] got shape {mask.shape}."
            raise ValueError(msg)
        if mask.ndim == 4:
            if mask.shape[1] != 1:
                msg = f"Mask must have 1 channel, got {mask.shape[1]}."
                raise ValueError(msg)
            mask = mask.squeeze(1)
        return Mask(mask, dtype=torch.bool)

    @staticmethod
    def validate_anomaly_map(anomaly_map: torch.Tensor | None) -> Mask | None:
        """Validate anomaly map.

        Validates and converts anomaly maps to float32 Mask objects.

        Args:
            anomaly_map (torch.Tensor | None): Input anomaly map tensor with shape either:
                - ``[T, H, W]`` for multiple frames
                - ``[T, 1, H, W]`` for multiple frames with channel dimension
                where ``H`` is height, ``W`` width, and ``T`` number of frames.

        Returns:
            Mask | None: Validated float32 mask or ``None``.

        Raises:
            TypeError: If ``anomaly_map`` is not a ``torch.Tensor`` or ``None``.
            ValueError: If anomaly map shape is invalid.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoValidator
            >>> # Multi-frame anomaly map
            >>> amap = torch.rand(10, 1, 224, 224)
            >>> validated = VideoValidator.validate_anomaly_map(amap)
            >>> isinstance(validated, Mask)
            True
            >>> validated.shape
            torch.Size([10, 224, 224])
        """
        if anomaly_map is None:
            return None
        if not isinstance(anomaly_map, torch.Tensor):
            msg = f"Anomaly map must be a torch.Tensor, got {type(anomaly_map)}."
            raise TypeError(msg)
        if anomaly_map.ndim not in {3, 4}:
            msg = f"Anomaly map must have shape [T, H, W] or [T, 1, H, W], got shape {anomaly_map.shape}."
            raise ValueError(msg)
        if anomaly_map.ndim == 4:
            if anomaly_map.shape[1] != 1:
                msg = f"Anomaly map with 4 dimensions must have 1 channel, got {anomaly_map.shape[1]}."
                raise ValueError(msg)
            anomaly_map = anomaly_map.squeeze(1)

        return Mask(anomaly_map, dtype=torch.float32)

    @staticmethod
    def validate_video_path(video_path: str | None) -> str | None:
        """Validate video file path.

        Args:
            video_path (str | None): Input video file path or ``None``.

        Returns:
            str | None: Validated video path or ``None``.

        Examples:
            >>> from anomalib.data.validators import VideoValidator
            >>> path = "/path/to/video.mp4"
            >>> validated = VideoValidator.validate_video_path(path)
            >>> validated == path
            True
        """
        return validate_path(video_path) if video_path else None

    @staticmethod
    def validate_mask_path(mask_path: str | None) -> str | None:
        """Validate mask file path.

        Args:
            mask_path (str | None): Input mask file path or ``None``.

        Returns:
            str | None: Validated mask path or ``None``.

        Examples:
            >>> from anomalib.data.validators import VideoValidator
            >>> path = "/path/to/mask.mp4"
            >>> validated = VideoValidator.validate_mask_path(path)
            >>> validated == path
            True
        """
        return validate_path(mask_path) if mask_path else None

    @staticmethod
    def validate_pred_score(
        pred_score: torch.Tensor | float | None,
        anomaly_map: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Validate prediction score.

        Validates prediction scores and optionally computes them from anomaly maps.

        Args:
            pred_score (torch.Tensor | float | None): Input prediction score or ``None``.
            anomaly_map (torch.Tensor | None): Optional anomaly map to compute score from.

        Returns:
            torch.Tensor | None: Validated float32 prediction score or ``None``.

        Raises:
            TypeError: If ``pred_score`` is not a float, tensor or ``None``.
            ValueError: If prediction score is not a scalar.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoValidator
            >>> score = 0.8
            >>> validated = VideoValidator.validate_pred_score(score)
            >>> validated
            tensor(0.8000)
        """
        if pred_score is None:
            return torch.amax(anomaly_map, dim=(-3, -2, -1)) if anomaly_map is not None else None

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
        """Validate prediction mask.

        Args:
            pred_mask (torch.Tensor | None): Input prediction mask tensor or ``None``.

        Returns:
            Mask | None: Validated prediction mask or ``None``.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoValidator
            >>> mask = torch.randint(0, 2, (10, 1, 224, 224))
            >>> validated = VideoValidator.validate_pred_mask(mask)
            >>> isinstance(validated, Mask)
            True
            >>> validated.shape
            torch.Size([10, 224, 224])
        """
        return VideoValidator.validate_gt_mask(pred_mask)  # We can reuse the gt_mask validation

    @staticmethod
    def validate_pred_label(pred_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate prediction label.

        Args:
            pred_label (torch.Tensor | None): Input prediction label or ``None``.

        Returns:
            torch.Tensor | None: Validated boolean prediction label or ``None``.

        Raises:
            TypeError: If ``pred_label`` is not a ``torch.Tensor``.
            ValueError: If prediction label is not a scalar.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoValidator
            >>> label = torch.tensor(1)
            >>> validated = VideoValidator.validate_pred_label(label)
            >>> validated
            tensor(True)
        """
        if pred_label is None:
            return None
        if not isinstance(pred_label, torch.Tensor):
            try:
                pred_label = torch.tensor(pred_label)
            except Exception as e:
                msg = "Failed to convert pred_label to a torch.Tensor."
                raise ValueError(msg) from e
        pred_label = pred_label.squeeze()
        if pred_label.ndim != 0:
            msg = f"Predicted label must be a scalar, got shape {pred_label.shape}."
            raise ValueError(msg)
        return pred_label.to(torch.bool)

    @staticmethod
    def validate_original_image(original_image: torch.Tensor | Video | None) -> torch.Tensor | Video | None:
        """Validate original video or image.

        Args:
            original_image (torch.Tensor | Video | None): Input original video/image or
                ``None``.

        Returns:
            torch.Tensor | Video | None: Validated original video/image or ``None``.

        Raises:
            TypeError: If input is not a ``torch.Tensor`` or ``Video``.
            ValueError: If tensor shape is invalid.

        Examples:
            >>> import torch
            >>> from torchvision.tv_tensors import Video
            >>> from anomalib.data.validators import VideoValidator
            >>> # Video tensor
            >>> video = Video(torch.rand(10, 3, 224, 224))
            >>> validated = VideoValidator.validate_original_image(video)
            >>> validated.shape
            torch.Size([10, 3, 224, 224])

            >>> # Single image
            >>> image = torch.rand(3, 256, 256)
            >>> validated = VideoValidator.validate_original_image(image)
            >>> validated.shape
            torch.Size([3, 256, 256])
        """
        if original_image is None:
            return None

        if not isinstance(original_image, torch.Tensor | Video):
            msg = f"Original image must be a torch.Tensor or torchvision Video object, got {type(original_image)}."
            raise TypeError(msg)

        if original_image.ndim == 3:
            # Single image case
            if original_image.shape[0] != 3:
                msg = f"Original image must have 3 channels, got {original_image.shape[0]}."
                raise ValueError(msg)
        elif original_image.ndim == 4:
            # Video case
            if original_image.shape[1] != 3:
                msg = f"Original video must have 3 channels, got {original_image.shape[1]}."
                raise ValueError(msg)
        else:
            msg = f"Original image/video must have shape [C, H, W] or [T, C, H, W], got shape {original_image.shape}."
            raise ValueError(msg)

        return original_image

    @staticmethod
    def validate_target_frame(target_frame: int | None) -> int | None:
        """Validate target frame index.

        Args:
            target_frame (int | None): Input target frame index or ``None``.

        Returns:
            int | None: Validated target frame index or ``None``.

        Raises:
            TypeError: If ``target_frame`` is not an integer.
            ValueError: If target frame index is negative.

        Examples:
            >>> from anomalib.data.validators import VideoValidator
            >>> validated = VideoValidator.validate_target_frame(31)
            >>> print(validated)
            31
        """
        if target_frame is None:
            return None
        if not isinstance(target_frame, int):
            msg = f"Target frame must be an integer, got {type(target_frame)}."
            raise TypeError(msg)
        if target_frame < 0:
            msg = f"Target frame index must be non-negative, got {target_frame}."
            raise ValueError(msg)
        return target_frame

    @staticmethod
    def validate_frames(frames: torch.Tensor | None) -> torch.Tensor | None:
        """Validate frames tensor.

        Args:
            frames (torch.Tensor | None): Input frames tensor or frame indices or ``None``.

        Returns:
            torch.Tensor | None: Validated frames tensor or ``None``.

        Raises:
            TypeError: If ``frames`` is not a ``torch.Tensor``.
            ValueError: If frames tensor is not a 1D tensor of indices.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoValidator
            >>> indices = torch.tensor([0, 5, 10])
            >>> validated = VideoValidator.validate_frames(indices)
            >>> validated
            tensor([0, 5, 10])
        """
        if frames is None:
            return None
        if not isinstance(frames, torch.Tensor):
            msg = f"Frames must be a torch.Tensor, got {type(frames)}."
            raise TypeError(msg)

        # Ensure frames is a 1D tensor of indices
        if frames.ndim != 1 and frames.numel() != 1:
            msg = f"Frames must be a 1D tensor of indices or a single scalar tensor, got shape {frames.shape}."
            raise ValueError(msg)
        if frames.numel() == 1:
            frames = frames.view(1)
        # Ensure all indices are non-negative integers
        if not torch.all(frames >= 0) or not frames.dtype.is_floating_point:
            if not frames.dtype.is_floating_point:
                frames = frames.to(torch.int64)
            else:
                msg = "All frame indices must be non-negative integers."
                raise ValueError(msg)
        return frames

    @staticmethod
    def validate_last_frame(last_frame: torch.Tensor | int | float | None) -> torch.Tensor | int | None:
        """Validate last frame index.

        Args:
            last_frame (torch.Tensor | int | float | None): Input last frame index or
                ``None``.

        Returns:
            torch.Tensor | int | None: Validated last frame index or ``None``.

        Raises:
            TypeError: If ``last_frame`` is not a tensor, int, or float.
            ValueError: If last frame index is negative.

        Examples:
            >>> from anomalib.data.validators import VideoValidator
            >>> # Integer input
            >>> validated = VideoValidator.validate_last_frame(5)
            >>> print(validated)
            5

            >>> # Float input
            >>> validated = VideoValidator.validate_last_frame(5.7)
            >>> print(validated)
            5

            >>> # Tensor input
            >>> import torch
            >>> tensor_frame = torch.tensor(10.3)
            >>> validated = VideoValidator.validate_last_frame(tensor_frame)
            >>> print(validated)
            tensor(10)
        """
        if last_frame is None:
            return None
        if isinstance(last_frame, int | float):
            last_frame = int(last_frame)
            if last_frame < 0:
                msg = f"Last frame index must be non-negative, got {last_frame}."
                raise ValueError(msg)
            return last_frame
        if isinstance(last_frame, torch.Tensor):
            if last_frame.numel() != 1:
                msg = f"Last frame must be a scalar tensor, got shape {last_frame.shape}."
                raise ValueError(msg)
            last_frame = last_frame.int()
            if last_frame.item() < 0:
                msg = f"Last frame index must be non-negative, got {last_frame.item()}."
                raise ValueError(msg)
            return last_frame
        msg = f"Last frame must be an int, float, or a torch.Tensor, got {type(last_frame)}."
        raise TypeError(msg)

    @staticmethod
    def validate_explanation(explanation: str | None) -> str | None:
        """Validate the explanation string."""
        return ImageValidator.validate_explanation(explanation)


class VideoBatchValidator:
    """Validate ``torch.Tensor`` data for video batches.

    This class provides static methods to validate various video batch data types including
    tensors, masks, labels, paths and more. Each method performs thorough validation of
    its input and returns the validated data in the correct format.
    """

    @staticmethod
    def validate_image(image: Video) -> Video:
        """Validate the video batch tensor.

        Validates that the input video batch tensor has the correct dimensions, number of
        channels and data type. Converts the tensor to float32 and scales values to [0,1]
        range.

        Args:
            image (Video): Input video batch tensor. Should be either:
                - Shape ``(B,C,H,W)`` for single frame images
                - Shape ``(B,T,C,H,W)`` for multi-frame videos
                Where:
                    - ``B`` is batch size
                    - ``T`` is number of frames
                    - ``C`` is number of channels (1 or 3)
                    - ``H`` is height
                    - ``W`` is width

        Returns:
            Video: Validated and normalized video batch tensor.

        Raises:
            TypeError: If ``image`` is not a ``torch.Tensor``.
            ValueError: If tensor dimensions or channel count are invalid.

        Examples:
            >>> import torch
            >>> from torchvision.tv_tensors import Video
            >>> from anomalib.data.validators import VideoBatchValidator
            >>> # Create sample video batch with 2 videos, 10 frames each
            >>> video_batch = Video(torch.rand(2, 10, 3, 224, 224))
            >>> validated_batch = VideoBatchValidator.validate_image(video_batch)
            >>> print(validated_batch.shape)
            torch.Size([2, 10, 3, 224, 224])
        """
        if not isinstance(image, torch.Tensor):
            msg = f"Video batch must be a torch.Tensor, got {type(image)}."
            raise TypeError(msg)

        if image.dim() not in {4, 5}:  # (B, C, H, W) or (B, T, C, H, W)
            msg = (
                "Video batch must have 4 dimensions (B, C, H, W) for single frame images "
                f"or 5 dimensions (B, T, C, H, W) for multi-frame videos, got {image.dim()}."
            )
            raise ValueError(msg)

        if image.dim() == 5 and image.shape[2] not in {1, 3}:
            msg = f"Video batch must have 1 or 3 channels, got {image.shape[2]}."
            raise ValueError(msg)
        if image.dim() == 4 and image.shape[1] not in {1, 3}:
            msg = f"Image batch must have 1 or 3 channels, got {image.shape[1]}."
            raise ValueError(msg)

        return to_dtype_image(image, torch.float32, scale=True)

    @staticmethod
    def validate_gt_label(label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the ground truth labels for a batch.

        Validates that the input ground truth labels have the correct data type and
        format. Converts labels to boolean type.

        Args:
            label (torch.Tensor | None): Input ground truth labels. Should be a 1D tensor
                of boolean or integer values.

        Returns:
            torch.Tensor | None: Validated ground truth labels as boolean tensor.

        Raises:
            TypeError: If ``label`` is not a ``torch.Tensor`` or has invalid dtype.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoBatchValidator
            >>> gt_labels = torch.tensor([0, 1, 1, 0])
            >>> validated_labels = VideoBatchValidator.validate_gt_label(gt_labels)
            >>> print(validated_labels)
            tensor([False,  True,  True, False])
        """
        if label is None:
            return None
        if not isinstance(label, torch.Tensor):
            msg = f"Ground truth labels must be a torch.Tensor, got {type(label)}."
            raise TypeError(msg)
        if torch.is_floating_point(label):
            msg = f"Ground truth labels must be boolean or integer, got {label.dtype}."
            raise TypeError(msg)
        return label.bool()

    @staticmethod
    def validate_gt_mask(mask: torch.Tensor | None) -> Mask | None:
        """Validate the ground truth masks for a batch.

        Validates that the input ground truth masks have the correct shape and format.
        Converts masks to boolean type.

        Args:
            mask (torch.Tensor | None): Input ground truth masks. Should be one of:
                - Shape ``(H,W)`` for single mask
                - Shape ``(N,H,W)`` for batch of masks
                - Shape ``(N,1,H,W)`` for batch with channel dimension

        Returns:
            Mask | None: Validated ground truth masks as boolean tensor.

        Raises:
            TypeError: If ``mask`` is not a ``torch.Tensor``.
            ValueError: If mask shape is invalid.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoBatchValidator
            >>> # Create 10 frame masks
            >>> gt_masks = torch.rand(10, 224, 224) > 0.5
            >>> validated_masks = VideoBatchValidator.validate_gt_mask(gt_masks)
            >>> print(validated_masks.shape)
            torch.Size([10, 224, 224])
            >>> # Create 4 single-frame masks
            >>> single_frame_masks = torch.rand(4, 456, 256) > 0.5
            >>> validated_single_frame = VideoBatchValidator.validate_gt_mask(
            ...     single_frame_masks
            ... )
            >>> print(validated_single_frame.shape)
            torch.Size([4, 456, 256])
        """
        if mask is None:
            return None
        if not isinstance(mask, torch.Tensor):
            msg = f"Ground truth mask must be a torch.Tensor, got {type(mask)}."
            raise TypeError(msg)
        if mask.ndim not in {2, 3, 4}:
            msg = f"Ground truth mask must have shape [H, W] or [N, H, W] or [N, 1, H, W] got shape {mask.shape}."
            raise ValueError(msg)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 4:
            if mask.shape[1] != 1:
                msg = f"Ground truth mask must have 1 channel, got {mask.shape[1]}."
                raise ValueError(msg)
            mask = mask.squeeze(1)
        return Mask(mask, dtype=torch.bool)

    @staticmethod
    def validate_mask_path(mask_path: list[str] | None) -> list[str] | None:
        """Validate the mask paths for a batch.

        Validates that the input mask paths are in the correct format.

        Args:
            mask_path (list[str] | None): Input mask paths. Should be a list of strings
                containing valid file paths.

        Returns:
            list[str] | None: Validated mask paths.

        Raises:
            TypeError: If ``mask_path`` is not a list of strings.

        Examples:
            >>> from anomalib.data.validators import VideoBatchValidator
            >>> mask_paths = ["path/to/mask1.png", "path/to/mask2.png"]
            >>> validated_paths = VideoBatchValidator.validate_mask_path(mask_paths)
            >>> print(validated_paths)
            ['path/to/mask1.png', 'path/to/mask2.png']
        """
        return validate_batch_path(mask_path)

    @staticmethod
    def validate_anomaly_map(anomaly_map: torch.Tensor | None) -> Mask | None:
        """Validate the anomaly maps for a batch.

        Validates that the input anomaly maps have the correct shape and format.
        Converts maps to float32 type.

        Args:
            anomaly_map (torch.Tensor | None): Input anomaly maps. Should be either:
                - Shape ``(B,T,H,W)`` for single channel maps
                - Shape ``(B,T,1,H,W)`` for explicit single channel
                Where:
                    - ``B`` is batch size
                    - ``T`` is number of frames
                    - ``H`` is height
                    - ``W`` is width

        Returns:
            Mask | None: Validated anomaly maps as float32 tensor.

        Raises:
            TypeError: If ``anomaly_map`` is not a ``torch.Tensor``.
            ValueError: If anomaly map shape is invalid.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoBatchValidator
            >>> # Create maps for 2 videos with 10 frames each
            >>> anomaly_maps = torch.rand(2, 10, 224, 224)
            >>> validated_maps = VideoBatchValidator.validate_anomaly_map(anomaly_maps)
            >>> print(validated_maps.shape)
            torch.Size([2, 10, 224, 224])
        """
        if anomaly_map is None:
            return None
        if not isinstance(anomaly_map, torch.Tensor):
            msg = f"Anomaly maps must be a torch.Tensor, got {type(anomaly_map)}."
            raise TypeError(msg)
        if anomaly_map.ndim not in {4, 5}:
            msg = f"Anomaly maps must have shape [B, T, H, W] or [B, T, 1, H, W], got shape {anomaly_map.shape}."
            raise ValueError(msg)
        if anomaly_map.ndim == 5:
            if anomaly_map.shape[2] != 1:
                msg = f"Anomaly maps must have 1 channel, got {anomaly_map.shape[2]}."
                raise ValueError(msg)
            anomaly_map = anomaly_map.squeeze(2)
        return Mask(anomaly_map, dtype=torch.float32)

    @staticmethod
    def validate_pred_score(
        pred_score: torch.Tensor | None,
        anomaly_map: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Validate the prediction scores for a batch.

        Validates that the input prediction scores have the correct format. If no scores
        are provided but an anomaly map is given, computes scores from the map.

        Args:
            pred_score (torch.Tensor | None): Input prediction scores. Should be a 1D
                tensor of float values.
            anomaly_map (torch.Tensor | None, optional): Input anomaly map used to compute
                scores if ``pred_score`` is None.

        Returns:
            torch.Tensor | None: Validated prediction scores as float32 tensor.

        Raises:
            ValueError: If prediction scores have invalid shape or cannot be converted to
                tensor.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoBatchValidator
            >>> pred_scores = torch.tensor([0.1, 0.9, 0.3, 0.7])
            >>> validated_scores = VideoBatchValidator.validate_pred_score(pred_scores)
            >>> print(validated_scores)
            tensor([0.1000, 0.9000, 0.3000, 0.7000])
        """
        if pred_score is None:
            return torch.amax(anomaly_map, dim=(-3, -2, -1)) if anomaly_map is not None else None

        if not isinstance(pred_score, torch.Tensor):
            try:
                pred_score = torch.tensor(pred_score)
            except Exception as e:
                msg = "Failed to convert pred_score to a torch.Tensor."
                raise ValueError(msg) from e
        if pred_score.ndim != 1:
            msg = f"Predicted scores must be a 1D tensor, got shape {pred_score.shape}."
            raise ValueError(msg)

        return pred_score.to(torch.float32)

    @staticmethod
    def validate_pred_mask(pred_mask: torch.Tensor | None) -> Mask | None:
        """Validate the prediction masks for a batch.

        Validates prediction masks using the same logic as ground truth masks.

        Args:
            pred_mask (torch.Tensor | None): Input prediction masks. Should follow same
                format as ground truth masks.

        Returns:
            Mask | None: Validated prediction masks.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoBatchValidator
            >>> # Create masks for 2 videos with 10 frames each
            >>> pred_masks = torch.rand(2, 10, 224, 224) > 0.5
            >>> validated_masks = VideoBatchValidator.validate_pred_mask(pred_masks)
            >>> print(validated_masks.shape)
            torch.Size([2, 10, 224, 224])
        """
        return VideoBatchValidator.validate_gt_mask(pred_mask)  # Reuse gt_mask validation

    @staticmethod
    def validate_pred_label(pred_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction labels for a batch.

        Validates that the input prediction labels have the correct format and converts
        them to boolean type.

        Args:
            pred_label (torch.Tensor | None): Input prediction labels. Should be a 1D
                tensor of boolean or numeric values.

        Returns:
            torch.Tensor | None: Validated prediction labels as boolean tensor.

        Raises:
            ValueError: If prediction labels have invalid shape or cannot be converted to
                tensor.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoBatchValidator
            >>> pred_labels = torch.tensor([0, 1, 1, 0])
            >>> validated_labels = VideoBatchValidator.validate_pred_label(pred_labels)
            >>> print(validated_labels)
            tensor([False,  True,  True, False])
        """
        if pred_label is None:
            return None
        if not isinstance(pred_label, torch.Tensor):
            try:
                pred_label = torch.tensor(pred_label)
            except Exception as e:
                msg = "Failed to convert pred_label to a torch.Tensor."
                raise ValueError(msg) from e
        if pred_label.ndim != 1:
            msg = f"Predicted labels must be a 1D tensor, got shape {pred_label.shape}."
            raise ValueError(msg)
        return pred_label.to(torch.bool)

    @staticmethod
    def validate_original_image(original_image: torch.Tensor | Video | None) -> torch.Tensor | Video | None:
        """Validate the original videos for a batch.

        Validates that the input videos have the correct dimensions and channel count.
        Adds temporal dimension to single frame inputs.

        Args:
            original_image (torch.Tensor | Video | None): Input original videos. Should be
                either:
                - Shape ``(B,C,H,W)`` for single frame images
                - Shape ``(B,T,C,H,W)`` for multi-frame videos
                Where:
                    - ``B`` is batch size
                    - ``T`` is number of frames
                    - ``C`` is number of channels (must be 3)
                    - ``H`` is height
                    - ``W`` is width

        Returns:
            torch.Tensor | Video | None: Validated original videos.

        Raises:
            TypeError: If input is not a ``torch.Tensor`` or ``torchvision.Video``.
            ValueError: If video has invalid shape or channel count.

        Examples:
            >>> import torch
            >>> from torchvision.tv_tensors import Video
            >>> from anomalib.data.validators import VideoBatchValidator
            >>> # Create 2 videos with 10 frames each
            >>> original_videos = Video(torch.rand(2, 10, 3, 224, 224))
            >>> validated_videos = VideoBatchValidator.validate_original_image(
            ...     original_videos
            ... )
            >>> print(validated_videos.shape)
            torch.Size([2, 10, 3, 224, 224])
        """
        if original_image is None:
            return None

        if not isinstance(original_image, torch.Tensor | Video):
            msg = f"Original image must be a torch.Tensor or torchvision Video object, got {type(original_image)}."
            raise TypeError(msg)

        if original_image.ndim not in {4, 5}:  # (B, C, H, W) or (B, T, C, H, W)
            msg = (
                "Original image/video must have shape [B, C, H, W] for single frame or "
                f"[B, T, C, H, W] for multi-frame, got shape {original_image.shape}."
            )
            raise ValueError(msg)

        if original_image.ndim == 4:
            # Add a temporal dimension for single frame videos
            original_image = original_image.unsqueeze(1)
        if original_image.shape[2] != 3:
            msg = f"Original video must have 3 channels, got {original_image.shape[2]}."
            raise ValueError(msg)

        return original_image

    @staticmethod
    def validate_video_path(video_path: list[str] | None) -> list[str] | None:
        """Validate the video paths for a batch.

        Validates that the input video paths are in the correct format.

        Args:
            video_path (list[str] | None): Input video paths. Should be a list of strings
                containing valid file paths.

        Returns:
            list[str] | None: Validated video paths.

        Raises:
            TypeError: If ``video_path`` is not a list of strings.

        Examples:
            >>> from anomalib.data.validators import VideoBatchValidator
            >>> video_paths = ["path/to/video1.mp4", "path/to/video2.mp4"]
            >>> validated_paths = VideoBatchValidator.validate_video_path(video_paths)
            >>> print(validated_paths)
            ['path/to/video1.mp4', 'path/to/video2.mp4']
        """
        return validate_batch_path(video_path)

    @staticmethod
    def validate_target_frame(target_frame: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the target frame indices for a batch.

        Validates that the input target frame indices are non-negative integers.

        Args:
            target_frame (torch.Tensor | None): Input target frame indices. Should be a
                1D tensor of non-negative integers.

        Returns:
            torch.Tensor | None: Validated target frame indices as int64 tensor.

        Raises:
            TypeError: If ``target_frame`` is not a ``torch.Tensor``.
            ValueError: If target frame indices are invalid.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoBatchValidator
            >>> target_frames = torch.tensor([5, 8, 3, 7])
            >>> validated_frames = VideoBatchValidator.validate_target_frame(target_frames)
            >>> print(validated_frames)
            tensor([5, 8, 3, 7])
        """
        if target_frame is None:
            return None
        if not isinstance(target_frame, torch.Tensor):
            msg = f"Target frame must be a torch.Tensor, got {type(target_frame)}."
            raise TypeError(msg)
        if target_frame.ndim != 1:
            msg = f"Target frame must be a 1D tensor, got shape {target_frame.shape}."
            raise ValueError(msg)
        if not torch.all(target_frame >= 0):
            msg = "Target frame indices must be non-negative."
            raise ValueError(msg)
        return target_frame.to(torch.int64)

    @staticmethod
    def validate_frames(frames: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the frame indices for a batch.

        Validates that the input frame indices are non-negative integers and converts
        them to the correct shape.

        Args:
            frames (torch.Tensor | None): Input frame indices. Should be either:
                - Shape ``(N,)`` for 1D tensor
                - Shape ``(N,1)`` for 2D tensor

        Returns:
            torch.Tensor | None: Validated frame indices as 1D int64 tensor.

        Raises:
            TypeError: If ``frames`` is not a ``torch.Tensor``.
            ValueError: If frame indices are invalid.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoBatchValidator
            >>> frame_indices = torch.tensor([[0], [1], [2], [3], [4], [5]])
            >>> validated_indices = VideoBatchValidator.validate_frames(frame_indices)
            >>> print(validated_indices)
            tensor([0, 1, 2, 3, 4, 5])
        """
        if frames is None:
            return None
        if not isinstance(frames, torch.Tensor):
            msg = f"Frames must be a torch.Tensor, got {type(frames)}."
            raise TypeError(msg)
        if frames.ndim == 2 and frames.shape[1] == 1:
            frames = frames.squeeze(1)
        if frames.ndim != 1:
            msg = f"Frames must be a 1D tensor or a 2D tensor with shape (N, 1), got shape {frames.shape}."
            raise ValueError(msg)
        if not torch.all(frames >= 0):
            msg = "All frame indices must be non-negative."
            raise ValueError(msg)
        return frames.to(torch.int64)

    @staticmethod
    def validate_last_frame(last_frame: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the last frame indices for a batch.

        Validates that the input last frame indices are non-negative integers.

        Args:
            last_frame (torch.Tensor | None): Input last frame indices. Should be a 1D
                tensor of non-negative numeric values.

        Returns:
            torch.Tensor | None: Validated last frame indices as int64 tensor.

        Raises:
            TypeError: If ``last_frame`` is not a ``torch.Tensor``.
            ValueError: If last frame indices are invalid.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoBatchValidator
            >>> last_frames = torch.tensor([9.5, 12.2, 15.8, 10.0])
            >>> validated_last_frames = VideoBatchValidator.validate_last_frame(
            ...     last_frames
            ... )
            >>> print(validated_last_frames)
            tensor([ 9, 12, 15, 10])
        """
        if last_frame is None:
            return None
        if not isinstance(last_frame, torch.Tensor):
            msg = f"Last frame must be a torch.Tensor, got {type(last_frame)}."
            raise TypeError(msg)
        if last_frame.ndim != 1:
            msg = f"Last frame must be a 1D tensor, got shape {last_frame.shape}."
            raise ValueError(msg)
        last_frame = last_frame.int()
        if not torch.all(last_frame >= 0):
            msg = "Last frame indices must be non-negative."
            raise ValueError(msg)
        return last_frame

    @staticmethod
    def validate_explanation(explanation: list[str] | None) -> list[str] | None:
        """Validate the explanation string."""
        return ImageBatchValidator.validate_explanation(explanation)
