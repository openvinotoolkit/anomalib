"""Video utilities for processing video data in anomaly detection.

This module provides utilities for:

- Indexing video clips and their corresponding masks
- Converting videos between different codecs
- Handling video frames and clips in PyTorch format

Example:
    >>> from anomalib.data.utils.video import ClipsIndexer
    >>> # Create indexer for video files and masks
    >>> indexer = ClipsIndexer(
    ...     video_paths=["video1.mp4", "video2.mp4"],
    ...     mask_paths=["mask1.mp4", "mask2.mp4"],
    ...     clip_length_in_frames=16
    ... )
    >>> # Get video clip with metadata
    >>> video_item = indexer.get_item(0)
    >>> video_item.image.shape  # (16, 3, H, W)
    torch.Size([16, 3, 256, 256])
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import torch
from torchvision.datasets.video_utils import VideoClips

from anomalib.data import VideoItem


class ClipsIndexer(VideoClips, ABC):
    """Extension of torchvision's VideoClips class for video and mask indexing.

    This class extends ``VideoClips`` to handle both video frames and their
    corresponding mask annotations. It provides functionality to:

    - Index and retrieve video clips
    - Access corresponding mask frames
    - Track frame indices and video metadata

    Subclasses must implement the ``get_mask`` method. The default implementation
    assumes ``video_paths`` contains video files. For custom data formats
    (e.g., image sequences), subclasses should override ``get_clip`` and
    ``_compute_frame_pts``.

    Args:
        video_paths: List of paths to video files in the dataset
        mask_paths: List of paths to mask files corresponding to each video
        clip_length_in_frames: Number of frames in each clip. Defaults to ``2``
        frames_between_clips: Stride between consecutive clips. Defaults to ``1``
    """

    def __init__(
        self,
        video_paths: list[str],
        mask_paths: list[str],
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
    ) -> None:
        super().__init__(
            video_paths=video_paths,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            output_format="TCHW",
        )
        self.mask_paths = mask_paths

    def last_frame_idx(self, video_idx: int) -> int:
        """Get index of the last frame in a video.

        Args:
            video_idx: Index of the video in the dataset

        Returns:
            Index of the last frame
        """
        return self.clips[video_idx][-1][-1].item()

    @abstractmethod
    def get_mask(self, idx: int) -> torch.Tensor | None:
        """Get masks for the clip at the given index.

        Args:
            idx: Index of the clip

        Returns:
            Tensor containing mask frames, or None if no masks exist
        """
        raise NotImplementedError

    def get_item(self, idx: int) -> VideoItem:
        """Get video clip and metadata at the given index.

        Args:
            idx: Index of the clip to retrieve

        Returns:
            VideoItem containing the clip frames, masks, path and metadata
        """
        with warnings.catch_warnings():
            # silence warning caused by bug in torchvision
            # see https://github.com/pytorch/vision/issues/5787
            warnings.simplefilter("ignore")
            clip, _, _, _ = self.get_clip(idx)

        video_idx, clip_idx = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx]

        return VideoItem(
            image=clip,
            gt_mask=self.get_mask(idx),
            video_path=video_path,
            frames=clip_pts,
            last_frame=self.last_frame_idx(video_idx),
        )


def convert_video(input_path: Path, output_path: Path, codec: str = "MP4V") -> None:
    """Convert a video file to use a different codec.

    Creates the output directory if it doesn't exist. Reads the input video
    frame by frame and writes to a new file using the specified codec.

    Args:
        input_path: Path to the input video file
        output_path: Path where the converted video will be saved
        codec: FourCC code for the desired output codec. Defaults to ``"MP4V"``
    """
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    # create video reader for input file
    video_reader = cv2.VideoCapture(str(input_path))

    # create video writer for output file
    fourcc = cv2.VideoWriter_fourcc(*codec)
    frame_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frame_width, frame_height),
    )

    # read frames
    success, frame = video_reader.read()
    while success:
        video_writer.write(frame)
        success, frame = video_reader.read()

    video_reader.release()
    video_writer.release()
