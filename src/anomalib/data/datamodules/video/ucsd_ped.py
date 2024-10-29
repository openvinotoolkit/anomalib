"""UCSD Pedestrian Data Module."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from shutil import move

from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data.datamodules.base.video import AnomalibVideoDataModule
from anomalib.data.datasets.base.video import VideoTargetFrame
from anomalib.data.datasets.video.ucsd_ped import UCSDpedDataset
from anomalib.data.utils import DownloadInfo, Split, ValSplitMode, download_and_extract

logger = logging.getLogger(__name__)

DOWNLOAD_INFO = DownloadInfo(
    name="UCSD Pedestrian",
    url="http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz",
    hashsum="2329af326951f5097fdd114c50e853957d3e569493a49d22fc082a9fd791915b",
)


class UCSDped(AnomalibVideoDataModule):
    """UCSDped DataModule class.

    Args:
        root (Path | str): Path to the root of the dataset
        category (str): Sub-category of the dataset, e.g. "UCSDped1" or "UCSDped2"
        clip_length_in_frames (int, optional): Number of video frames in each clip.
        frames_between_clips (int, optional): Number of frames between each consecutive video clip.
        target_frame (VideoTargetFrame): Specifies the target frame in the video clip, used for ground truth retrieval
        task (TaskType): Task type, 'classification', 'detection' or 'segmentation'
        train_batch_size (int, optional): Training batch size. Defaults to 32.
        eval_batch_size (int, optional): Test batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 8.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
    """

    def __init__(
        self,
        root: Path | str = "./datasets/ucsd",
        category: str = "UCSDped2",
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 10,
        target_frame: VideoTargetFrame = VideoTargetFrame.LAST,
        task: TaskType | str = TaskType.SEGMENTATION,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 8,
        val_split_mode: ValSplitMode = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.task = TaskType(task)
        self.root = Path(root)
        self.category = category

        self.clip_length_in_frames = clip_length_in_frames
        self.frames_between_clips = frames_between_clips
        self.target_frame = VideoTargetFrame(target_frame)

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = UCSDpedDataset(
            task=self.task,
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
            target_frame=self.target_frame,
            root=self.root,
            category=self.category,
            split=Split.TRAIN,
        )

        self.test_data = UCSDpedDataset(
            task=self.task,
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
            target_frame=self.target_frame,
            root=self.root,
            category=self.category,
            split=Split.TEST,
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available."""
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)

            # move contents to root
            extracted_folder = self.root / "UCSD_Anomaly_Dataset.v1p2"
            for filename in extracted_folder.glob("*"):
                move(str(filename), str(self.root / filename.name))
            extracted_folder.rmdir()
