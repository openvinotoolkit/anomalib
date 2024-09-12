"""Validate torch video data."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision.transforms.v2.functional import to_dtype_image
from torchvision.tv_tensors import Image, Mask, Video

from anomalib.data.validators.path import validate_path


class VideoValidator:
    """Validate torch.Tensor data for videos."""

    @staticmethod
    def validate_image(image: Image) -> Image:
        """Validate the video tensor.

        Args:
            image (Image): Input tensor.

        Returns:
            Image: Validated tensor.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the video tensor does not have the correct shape.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoValidator
            >>> video = torch.rand(10, 3, 256, 256)  # 10 frames, RGB
            >>> validated_video = VideoValidator.validate_image(video)
            >>> validated_video.shape
            torch.Size([10, 3, 256, 256])
            >>> single_frame_rgb = torch.rand(3, 256, 256)  # Single RGB frame
            >>> validated_single_frame_rgb = VideoValidator.validate_image(single_frame_rgb)
            >>> validated_single_frame_rgb.shape
            torch.Size([1, 3, 256, 256])
            >>> single_frame_gray = torch.rand(1, 256, 256)  # Single grayscale frame
            >>> validated_single_frame_gray = VideoValidator.validate_image(single_frame_gray)
            >>> validated_single_frame_gray.shape
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
            >>> from anomalib.data.validators import VideoValidator
            >>> label_int = 1
            >>> validated_label = VideoValidator.validate_gt_label(label_int)
            >>> validated_label
            tensor(True)
            >>> label_tensor = torch.tensor([0, 0], dtype=torch.int32)
            >>> validated_label = VideoValidator.validate_gt_label(label_tensor)
            >>> validated_label
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
            >>> from anomalib.data.validators import VideoValidator
            >>> mask = torch.randint(0, 2, (10, 1, 224, 224))  # 10 frames
            >>> validated_mask = VideoValidator.validate_gt_mask(mask)
            >>> isinstance(validated_mask, Mask)
            True
            >>> validated_mask.shape
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
        """Validate the anomaly map.

        Args:
            anomaly_map (torch.Tensor | None): Input anomaly map.

        Returns:
            Mask | None: Validated anomaly map as a Mask, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the anomaly map shape is invalid.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoValidator
            >>> anomaly_map = torch.rand(10, 1, 224, 224)  # 10 frames
            >>> validated_map = VideoValidator.validate_anomaly_map(anomaly_map)
            >>> isinstance(validated_map, Mask)
            True
            >>> validated_map.shape
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
        """Validate the video path.

        Args:
            video_path (str | None): Input video path.

        Returns:
            str | None: Validated video path, or None.

        Examples:
            >>> from anomalib.data.validators import VideoValidator
            >>> path = "/path/to/video.mp4"
            >>> validated_path = VideoValidator.validate_video_path(path)
            >>> validated_path == path
            True
        """
        return validate_path(video_path) if video_path else None

    @staticmethod
    def validate_mask_path(mask_path: str | None) -> str | None:
        """Validate the mask path.

        Args:
            mask_path (str | None): Input mask path.

        Returns:
            str | None: Validated mask path, or None.

        Examples:
            >>> from anomalib.data.validators import VideoValidator
            >>> path = "/path/to/mask.mp4"
            >>> validated_path = VideoValidator.validate_mask_path(path)
            >>> validated_path == path
            True
        """
        return validate_path(mask_path) if mask_path else None

    @staticmethod
    def validate_pred_score(
        pred_score: torch.Tensor | float | None,
        anomaly_map: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Validate the prediction score.

        Args:
            pred_score (torch.Tensor | float | None): Input prediction score.
            anomaly_map (torch.Tensor | None): Input anomaly map.

        Returns:
            torch.Tensor | None: Validated prediction score as a float32 tensor, or None.

        Raises:
            TypeError: If the input is neither a float, torch.Tensor, nor None.
            ValueError: If the prediction score is not a scalar.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoValidator
            >>> score = 0.8
            >>> validated_score = VideoValidator.validate_pred_score(score)
            >>> validated_score
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
        """Validate the prediction mask.

        Args:
            pred_mask (torch.Tensor | None): Input prediction mask.

        Returns:
            Mask | None: Validated prediction mask, or None.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoValidator
            >>> mask = torch.randint(0, 2, (10, 1, 224, 224))  # 10 frames
            >>> validated_mask = VideoValidator.validate_pred_mask(mask)
            >>> isinstance(validated_mask, Mask)
            True
            >>> validated_mask.shape
            torch.Size([10, 224, 224])
        """
        return VideoValidator.validate_gt_mask(pred_mask)  # We can reuse the gt_mask validation

    @staticmethod
    def validate_pred_label(pred_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction label.

        Args:
            pred_label (torch.Tensor | None): Input prediction label.

        Returns:
            torch.Tensor | None: Validated prediction label as a boolean tensor, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the prediction label is not a scalar.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoValidator
            >>> label = torch.tensor(1)
            >>> validated_label = VideoValidator.validate_pred_label(label)
            >>> validated_label
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
        """Validate the original video or image.

        Args:
            original_image (torch.Tensor | Video | None): Input original video or image.

        Returns:
            torch.Tensor | Video | None: Validated original video or image.

        Raises:
            TypeError: If the input is not a torch.Tensor or torchvision Video object.
            ValueError: If the tensor does not have the correct shape.

        Examples:
            >>> import torch
            >>> from torchvision.tv_tensors import Video
            >>> from anomalib.data.validators import VideoValidator
            >>> video = Video(torch.rand(10, 3, 224, 224))  # 10 frames
            >>> validated_video = VideoValidator.validate_original_image(video)
            >>> validated_video.shape
            torch.Size([10, 3, 224, 224])
            >>> image = torch.rand(3, 256, 256)  # Single image
            >>> validated_image = VideoValidator.validate_original_image(image)
            >>> validated_image.shape
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
        """Validate the target frame index.

        Args:
            target_frame (int | None): Input target frame index.

        Returns:
            int | None: Validated target frame index, or None.

        Raises:
            TypeError: If the input is not an integer.
            ValueError: If the target frame index is negative.

        Examples:
            >>> from anomalib.data.validators import VideoValidator
            >>> validated_frame = VideoValidator.validate_target_frame(31)
            >>> print(validated_frame)
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
        """Validate the frames tensor.

        Args:
            frames (torch.Tensor | None): Input frames tensor or frame indices.

        Returns:
            torch.Tensor | None: Validated frames tensor, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the frames tensor is not a 1D tensor of indices.

        Examples:
            >>> import torch
            >>> from anomalib.data.validators import VideoValidator
            >>> frame_indices = torch.tensor([0, 5, 10])
            >>> validated_indices = VideoValidator.validate_frames(frame_indices)
            >>> validated_indices
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
    def validate_last_frame(last_frame: torch.Tensor | int | None) -> torch.Tensor | int | None:
        """Validate the last frame index.

        Args:
            last_frame (torch.Tensor | int | None): Input last frame index.

        Returns:
            torch.Tensor | int | None: Validated last frame index, or None.

        Raises:
            TypeError: If the input is not a torch.Tensor or int.
            ValueError: If the last frame index is negative.

        Examples:
            >>> from anomalib.data.validators import VideoValidator
            >>> validated_frame = VideoValidator.validate_last_frame(5)
            >>> print(validated_frame)
            5
            >>> import torch
            >>> tensor_frame = torch.tensor(10)
            >>> validated_tensor = VideoValidator.validate_last_frame(tensor_frame)
            >>> print(validated_tensor)
            tensor(10)
        """
        if last_frame is None:
            return None
        if isinstance(last_frame, int):
            if last_frame < 0:
                msg = f"Last frame index must be non-negative, got {last_frame}."
                raise ValueError(msg)
            return last_frame
        if isinstance(last_frame, torch.Tensor):
            if last_frame.numel() != 1:
                msg = f"Last frame must be a scalar tensor, got shape {last_frame.shape}."
                raise ValueError(msg)
            if last_frame.item() < 0:
                msg = f"Last frame index must be non-negative, got {last_frame.item()}."
                raise ValueError(msg)
            return last_frame
        msg = f"Last frame must be an int or a torch.Tensor, got {type(last_frame)}."
        raise TypeError(msg)
