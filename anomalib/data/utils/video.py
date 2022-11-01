"""Video utils."""

import glob
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision.datasets.video_utils import VideoClips

from anomalib.data.utils import read_image


def read_frames_from_video(video_path: str, frame_idx: Iterable[int], image_size: Tuple[int, int] = None):
    """Read images from a folder of video frames."""
    frames = sorted(glob.glob(video_path + "/*"))

    frame_paths = [frames[pt] for pt in frame_idx]
    video = np.stack([read_image(frame_path) for frame_path in frame_paths])

    if image_size:
        height, width = image_size
        video = np.stack([cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA) for image in video])
    return video


class ClipsFromFolderMixin:
    """Mixin for video consisting of single-frame images arranged in folders.

    Files are expected to have this structure:
        path/to/dataset/train/video_0/frame_0.png
        path/to/dataset/train/video_0/frame_1.png
        path/to/dataset/train/video_0/frame_2.png
        ...
        path/to/dataset/train/video_1/frame_0.png
        path/to/dataset/train/video_1/frame_1.png
        path/to/dataset/train/video_1/frame_2.png
        ...
        path/to/dataset/train/video_n/frame_m.png
    """

    video_paths: List[str]
    num_clips: Callable
    get_clip_location: Callable
    clips: List[Tensor]

    def _compute_frame_pts(self) -> None:
        """Retrieve the number of frames in each video."""
        self.video_pts = []
        for video_path in self.video_paths:
            n_frames = len(glob.glob(video_path + "/*"))
            self.video_pts.append(Tensor(range(n_frames)))

        self.video_fps = [None] * len(self.video_paths)  # fps information cannot be inferred from folder structure

    def get_clip(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], int]:
        """Gets a subclip from a list of videos.

        Args:
            idx (int): index of the subclip. Must be between 0 and num_clips().

        Returns:
            video (Tensor)
            audio (Tensor)
            info (Dict)
            video_idx (int): index of the video in `video_paths`
        """
        if idx >= self.num_clips():
            raise IndexError(f"Index {idx} out of range ({self.num_clips()} number of clips)")
        video_idx, clip_idx = self.get_clip_location(idx)
        video_path = self.video_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx]

        frames = sorted(glob.glob(video_path + "/*"))

        frame_paths = [frames[pt] for pt in clip_pts.int()]
        video = torch.stack([Tensor(read_image(frame_path)) for frame_path in frame_paths])

        return video, torch.empty((1, 0)), {}, video_idx


class ClipsIndexer(VideoClips, ABC):
    """Extension of torchvision's VideoClips class that also returns the masks for each clip.

    Subclasses should implement the get_mask method. By default, the class inherits the functionality of VideoClips,
    which assumes that video_paths is a list of video files. If custom behaviour is required (e.g. video_paths is a list
    of folders with single-frame images), the subclass should implement at least get_clip and _compute_frame_pts.

    Args:
        video_paths (List[str]): List of video paths that make up the dataset.
        mask_paths (List[str]): List of paths to the masks for each video in the dataset.
    """

    def __init__(self, video_paths: List[str], mask_paths: List[str], *args, **kwargs) -> None:
        super().__init__(video_paths=video_paths, *args, **kwargs)
        self.mask_paths = mask_paths

    def last_frame_idx(self, video_idx: int) -> int:
        """Returns the index of the last frame for a given video."""
        return self.clips[video_idx][-1][-1].item()

    @abstractmethod
    def get_mask(self, idx) -> Optional[Tensor]:
        """Return the masks for the given index."""

    def get_item(self, idx: int) -> Dict[str, Any]:
        """Return a dictionary containing the clip, mask, video path and frame indices."""
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
