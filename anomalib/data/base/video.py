"""Base Video Dataset."""

from abc import ABC
from typing import Callable, Dict, Optional, Union

import torch
from torch import Tensor

from anomalib.data.base.dataset import AnomalibDataset
from anomalib.data.utils.video import ClipsIndexer
from anomalib.pre_processing import PreProcessor


class VideoAnomalibDataset(AnomalibDataset, ABC):
    """Base video anomalib dataset class.

    Args:
        task (str): Task type, either 'classification' or 'segmentation'
        pre_process (PreProcessor): Pre-processor object
        clip_length_in_frames (int): Number of video frames in each clip.
        frames_between_clips (int): Number of frames between each consecutive video clip.
    """

    def __init__(self, task: str, pre_process: PreProcessor, clip_length_in_frames: int, frames_between_clips: int):
        super().__init__(task, pre_process)

        self.clip_length_in_frames = clip_length_in_frames
        self.frames_between_clips = frames_between_clips
        self.pre_process = pre_process

        self.indexer: Optional[ClipsIndexer] = None
        self.indexer_cls: Optional[Callable] = None

    def __len__(self) -> int:
        """Get length of the dataset."""
        assert isinstance(self.indexer, ClipsIndexer)
        return self.indexer.num_clips()

    @property
    def samples(self):
        """Get the samples dataframe."""
        return super().samples

    @samples.setter
    def samples(self, samples):
        """Overwrite samples and re-index subvideos."""
        super(VideoAnomalibDataset, self.__class__).samples.fset(self, samples)
        self._setup_clips()

    def _setup_clips(self) -> None:
        """Compute the video and frame indices of the subvideos.

        Should be called after each change to self._samples
        """
        assert callable(self.indexer_cls)
        self.indexer = self.indexer_cls(  # pylint: disable=not-callable
            video_paths=list(self.samples.image_path),
            mask_paths=list(self.samples.mask_path),
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
        )

    def __getitem__(self, index: int) -> Dict[str, Union[str, Tensor]]:
        """Return mask, clip and file system information."""
        assert isinstance(self.indexer, ClipsIndexer)

        item = self.indexer.get_item(index)
        # include the untransformed image for visualization
        item["original_image"] = item["image"].to(torch.uint8)

        # apply transforms
        if "mask" in item and item["mask"] is not None:
            processed_frames = [
                self.pre_process(image=frame.numpy(), mask=mask) for frame, mask in zip(item["image"], item["mask"])
            ]
            item["image"] = torch.stack([item["image"] for item in processed_frames]).squeeze(0)
            mask = item["mask"]
            item["mask"] = torch.stack([item["mask"] for item in processed_frames]).squeeze(0)
            item["label"] = Tensor([1 in frame for frame in mask]).int().squeeze(0)
        else:
            item["image"] = torch.stack(
                [self.pre_process(image=frame.numpy())["image"] for frame in item["image"]]
            ).squeeze(0)

        if item["mask"] is None:
            item.pop("mask")

        return item
