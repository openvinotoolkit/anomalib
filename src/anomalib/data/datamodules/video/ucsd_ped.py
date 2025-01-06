"""UCSD Pedestrian Data Module.

This module provides a PyTorch Lightning data module for the UCSD Pedestrian dataset.
The dataset consists of surveillance videos of pedestrians, with anomalies defined as
non-pedestrian entities like cars, bikes, etc.
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from shutil import move

from torchvision.transforms.v2 import Transform

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
    """UCSD Pedestrian DataModule Class.

    Args:
        root (Path | str): Path to the root directory where the dataset will be
            downloaded and extracted. Defaults to ``"./datasets/ucsd"``.
        category (str): Dataset subcategory. Must be either ``"UCSDped1"`` or
            ``"UCSDped2"``. Defaults to ``"UCSDped2"``.
        clip_length_in_frames (int): Number of frames in each video clip.
            Defaults to ``2``.
        frames_between_clips (int): Number of frames between consecutive video
            clips. Defaults to ``10``.
        target_frame (VideoTargetFrame): Specifies which frame in the clip should
            be used for ground truth. Defaults to ``VideoTargetFrame.LAST``.
        train_batch_size (int): Batch size for training. Defaults to ``8``.
        eval_batch_size (int): Batch size for validation and testing.
            Defaults to ``8``.
        num_workers (int): Number of workers for data loading. Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply dto the training images
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
        val_split_mode (ValSplitMode): Determines how validation set is created.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of data to use for validation.
            Must be between 0 and 1. Defaults to ``0.5``.
        seed (int | None): Random seed for reproducibility. Defaults to ``None``.

    Example:
        >>> datamodule = UCSDped(root="./datasets/ucsd")
        >>> datamodule.setup()  # Downloads and prepares the dataset
        >>> train_loader = datamodule.train_dataloader()
        >>> val_loader = datamodule.val_dataloader()
        >>> test_loader = datamodule.test_dataloader()
    """

    def __init__(
        self,
        root: Path | str = "./datasets/ucsd",
        category: str = "UCSDped2",
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 10,
        target_frame: VideoTargetFrame = VideoTargetFrame.LAST,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        val_split_mode: ValSplitMode = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.root = Path(root)
        self.category = category

        self.clip_length_in_frames = clip_length_in_frames
        self.frames_between_clips = frames_between_clips
        self.target_frame = VideoTargetFrame(target_frame)

    def _setup(self, _stage: str | None = None) -> None:
        """Set up train and test datasets.

        Args:
            _stage (str | None): Stage for Lightning. Can be "fit" or "test".
        """
        self.train_data = UCSDpedDataset(
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
            target_frame=self.target_frame,
            root=self.root,
            category=self.category,
            split=Split.TRAIN,
        )

        self.test_data = UCSDpedDataset(
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
            target_frame=self.target_frame,
            root=self.root,
            category=self.category,
            split=Split.TEST,
        )

    def prepare_data(self) -> None:
        """Download and extract the dataset if not already available.

        The method checks if the dataset directory exists. If not, it downloads
        and extracts the dataset to the specified root directory.
        """
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)

            # move contents to root
            extracted_folder = self.root / "UCSD_Anomaly_Dataset.v1p2"
            for filename in extracted_folder.glob("*"):
                move(str(filename), str(self.root / filename.name))
            extracted_folder.rmdir()
