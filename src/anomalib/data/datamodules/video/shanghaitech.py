"""ShanghaiTech Campus Data Module.

This module provides a PyTorch Lightning DataModule for the ShanghaiTech Campus
dataset. If the dataset is not available locally, it will be downloaded and
extracted automatically. The video files are also converted to a format readable
by pyav.

Example:
    Create a ShanghaiTech datamodule::

        >>> from anomalib.data import ShanghaiTech
        >>> datamodule = ShanghaiTech(
        ...     root="./datasets/shanghaitech",
        ...     scene=1,
        ...     clip_length_in_frames=2,
        ...     frames_between_clips=1,
        ... )
        >>> datamodule.setup()
        >>> i, data = next(enumerate(datamodule.train_dataloader()))
        >>> data.keys()
        dict_keys(['image', 'video_path', 'frames', 'label'])

Notes:
    The directory structure after preparation will be::

        root/
        ├── testing/
        │   ├── frames/
        │   ├── test_frame_mask/
        │   └── test_pixel_mask/
        └── training/
            ├── frames/
            ├── converted_videos/
            └── videos/

License:
    ShanghaiTech Campus Dataset is released under the BSD 2-Clause License.

Reference:
    Liu, W., Luo, W., Lian, D., & Gao, S. (2018). Future frame prediction for
    anomaly detection--a new baseline. In Proceedings of the IEEE conference on
    computer vision and pattern recognition (pp. 6536-6545).
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from shutil import move

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.video import AnomalibVideoDataModule
from anomalib.data.datasets.base.video import VideoTargetFrame
from anomalib.data.datasets.video.shanghaitech import ShanghaiTechDataset
from anomalib.data.utils import DownloadInfo, Split, ValSplitMode, download_and_extract
from anomalib.data.utils.video import convert_video

logger = logging.getLogger(__name__)

DATASET_DOWNLOAD_INFO = DownloadInfo(
    name="ShanghaiTech Dataset",
    url="http://101.32.75.151:8181/dataset/shanghaitech.tar.gz",
    hashsum="c13a827043b259ccf8493c9d9130486872992153a9d714fe229e523cd4c94116",
)


class ShanghaiTech(AnomalibVideoDataModule):
    """ShanghaiTech DataModule class.

    Args:
        root (Path | str): Path to the root directory of the dataset.
            Defaults to ``"./datasets/shanghaitech"``.
        scene (int): Scene index in range [1, 13].
            Defaults to ``1``.
        clip_length_in_frames (int): Number of frames in each video clip.
            Defaults to ``2``.
        frames_between_clips (int): Number of frames between consecutive clips.
            Defaults to ``1``.
        target_frame (VideoTargetFrame): Specifies which frame in the clip should
            be used for ground truth.
            Defaults to ``VideoTargetFrame.LAST``.
        train_batch_size (int): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int): Test batch size.
            Defaults to ``32``.
        num_workers (int): Number of workers for data loading.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply dto the training images
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
        val_split_mode (ValSplitMode): Setting that determines how validation
            subset is obtained.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of train or test images that will be
            reserved for validation.
            Defaults to ``0.5``.
        seed (int | None): Random seed for reproducibility.
            Defaults to ``None``.
    """

    def __init__(
        self,
        root: Path | str = "./datasets/shanghaitech",
        scene: int = 1,
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
        target_frame: VideoTargetFrame = VideoTargetFrame.LAST,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
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
        self.scene = scene

        self.clip_length_in_frames = clip_length_in_frames
        self.frames_between_clips = frames_between_clips
        self.target_frame = target_frame

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = ShanghaiTechDataset(
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
            target_frame=self.target_frame,
            root=self.root,
            scene=self.scene,
            split=Split.TRAIN,
        )

        self.test_data = ShanghaiTechDataset(
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
            target_frame=self.target_frame,
            root=self.root,
            scene=self.scene,
            split=Split.TEST,
        )

    def prepare_data(self) -> None:
        """Download the dataset and convert video files."""
        training_root = self.root / "training"
        if training_root.is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DATASET_DOWNLOAD_INFO)

            # move contents to root
            extracted_folder = self.root / "shanghaitech"
            for filename in extracted_folder.glob("*"):
                move(str(filename), str(self.root / filename.name))
            extracted_folder.rmdir()

        # convert images if not done already
        vid_dir = training_root / "videos"
        converted_vid_dir = training_root / "converted_videos"
        vid_count = len(list(vid_dir.glob("*")))
        converted_vid_count = len(list(converted_vid_dir.glob("*")))
        if vid_count != converted_vid_count:
            self._convert_training_videos(vid_dir, converted_vid_dir)

    @staticmethod
    def _convert_training_videos(video_folder: Path, target_folder: Path) -> None:
        """Re-code training videos for correct frame reading by torchvision.

        The encoding of the raw video files in the ShanghaiTech dataset causes
        issues when reading frames using pyav. To prevent this, frames are read
        using opencv and written to new video files that can be parsed correctly
        with pyav.

        Args:
            video_folder (Path): Path to the folder containing training videos.
            target_folder (Path): Path where converted videos will be stored.
        """
        training_videos = sorted(video_folder.glob("*"))
        for video_idx, video_path in enumerate(training_videos):
            logger.info(
                "Converting training video %s (%i/%i)...",
                video_path.name,
                video_idx + 1,
                len(training_videos),
            )
            file_name = video_path.name
            target_path = target_folder / file_name
            convert_video(video_path, target_path, codec="XVID")
