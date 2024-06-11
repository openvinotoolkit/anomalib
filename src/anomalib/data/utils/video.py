"""Video utils."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import torch
from torchvision.datasets.video_utils import VideoClips


class ClipsIndexer(VideoClips, ABC):
    """Extension of torchvision's VideoClips class that also returns the masks for each clip.

    Subclasses should implement the get_mask method. By default, the class inherits the functionality of VideoClips,
    which assumes that video_paths is a list of video files. If custom behaviour is required (e.g. video_paths is a list
    of folders with single-frame images), the subclass should implement at least get_clip and _compute_frame_pts.

    Args:
        video_paths (list[str]): List of video paths that make up the dataset.
        mask_paths (list[str]): List of paths to the masks for each video in the dataset.
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
        """Return the index of the last frame for a given video."""
        return self.clips[video_idx][-1][-1].item()

    @abstractmethod
    def get_mask(self, idx: int) -> torch.Tensor | None:
        """Return the masks for the given index."""
        raise NotImplementedError

    def get_item(self, idx: int) -> dict[str, Any]:
        """Return a dictionary containing the clip, mask, video path and frame indices."""
        with warnings.catch_warnings():
            # silence warning caused by bug in torchvision, see https://github.com/pytorch/vision/issues/5787
            warnings.simplefilter("ignore")
            clip, _, _, _ = self.get_clip(idx)

        video_idx, clip_idx = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx]

        return {
            "image": clip,
            "mask": self.get_mask(idx),
            "video_path": video_path,
            "frames": clip_pts,
            "last_frame": self.last_frame_idx(video_idx),
        }


def convert_video(input_path: Path, output_path: Path, codec: str = "MP4V") -> None:
    """Convert video file to a different codec.

    Args:
        input_path (Path): Path to the input video.
        output_path (Path): Path to the target output video.
        codec (str): fourcc code of the codec that will be used for compression of the output file.
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
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    # read frames
    success, frame = video_reader.read()
    while success:
        video_writer.write(frame)
        success, frame = video_reader.read()

    video_reader.release()
    video_writer.release()
