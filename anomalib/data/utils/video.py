"""Video utils."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from torch import Tensor
from torchvision.datasets.video_utils import VideoClips


class ClipsIndexer(VideoClips, ABC):
    """Extension of torchvision's VideoClips class that also returns the masks for each clip.

    Subclasses should implement the get_mask method. By default, the class inherits the functionality of VideoClips,
    which assumes that video_paths is a list of video files. If custom behaviour is required (e.g. video_paths is a list
    of folders with single-frame images), the subclass should implement at least get_clip and _compute_frame_pts.

    Args:
        video_paths (List[str]): List of video paths that make up the dataset.
        mask_paths (List[str]): List of paths to the masks for each video in the dataset.
    """

    def __init__(
        self,
        video_paths: List[str],
        mask_paths: List[str],
        clip_length_in_frames: int = 1,
        frames_between_clips: int = 1,
    ) -> None:
        super().__init__(
            video_paths=video_paths,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
        )
        self.mask_paths = mask_paths

    def last_frame_idx(self, video_idx: int) -> int:
        """Returns the index of the last frame for a given video."""
        return self.clips[video_idx][-1][-1].item()

    @abstractmethod
    def get_mask(self, idx: int) -> Optional[Tensor]:
        """Return the masks for the given index."""
        raise NotImplementedError

    def get_item(self, idx: int) -> Dict[str, Any]:
        """Return a dictionary containing the clip, mask, video path and frame indices."""
        with warnings.catch_warnings():
            # silence warning caused by bug in torchvision, see https://github.com/pytorch/vision/issues/5787
            warnings.simplefilter("ignore")
            clip, _, _, _ = self.get_clip(idx)

        video_idx, clip_idx = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx]

        item = dict(
            image=clip,
            mask=self.get_mask(idx),
            video_path=video_path,
            frames=clip_pts,
            last_frame=self.last_frame_idx(video_idx),
        )

        return item
