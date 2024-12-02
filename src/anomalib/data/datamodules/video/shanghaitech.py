"""ShanghaiTech Campus Data Module.

Description:
    This module contains PyTorch Lightning DataModule for the ShanghaiTech Campus dataset.
    If the dataset is not on the file system, the DataModule class downloads and
    extracts the dataset and converts video files to a format that is readable by pyav.

License:
    ShanghaiTech Campus Dataset is released under the BSD 2-Clause License.

Reference:
    - W. Liu and W. Luo, D. Lian and S. Gao. "Future Frame Prediction for Anomaly Detection -- A New Baseline."
      IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2018.
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from shutil import move

from anomalib import TaskType
from anomalib.data.datamodules.base.video import AnomalibVideoDataModule
from anomalib.data.datasets.base.video import VideoTargetFrame
from anomalib.data.datasets.video.shanghaitech import ShanghaiTechDataset
from anomalib.data.utils import DownloadInfo, Split, ValSplitMode, download_and_extract
from anomalib.data.utils.split import SplitMode, resolve_split_mode
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
        root (Path | str): Path to the root of the dataset
        scene (int): Index of the dataset scene (category) in range [1, 13]
        clip_length_in_frames (int, optional): Number of video frames in each clip.
        frames_between_clips (int, optional): Number of frames between each consecutive video clip.
        target_frame (VideoTargetFrame): Specifies the target frame in the video clip, used for ground truth retrieval
        task TaskType): Task type, 'classification', 'detection' or 'segmentation'
        image_size (tuple[int, int], optional): Size to which input images should be resized.
            Defaults to ``None``.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        train_transform (Transform, optional): Transforms that should be applied to the input images during training.
            Defaults to ``None``.
        eval_transform (Transform, optional): Transforms that should be applied to the input images during evaluation.
            Defaults to ``None``.
        train_batch_size (int, optional): Training batch size. Defaults to 32.
        eval_batch_size (int, optional): Test batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 8.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
    """

    def __init__(
        self,
        root: Path | str = "./datasets/shanghaitech",
        scene: int = 1,
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
        target_frame: VideoTargetFrame = VideoTargetFrame.LAST,
        task: TaskType | str = TaskType.SEGMENTATION,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        val_split_mode: SplitMode | ValSplitMode | str = SplitMode.AUTO,
        val_split_ratio: float | None = None,
        seed: int | None = None,
    ) -> None:
        val_split_mode = resolve_split_mode(val_split_mode)
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
        self.scene = scene

        self.clip_length_in_frames = clip_length_in_frames
        self.frames_between_clips = frames_between_clips
        self.target_frame = target_frame

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = ShanghaiTechDataset(
            task=self.task,
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
            target_frame=self.target_frame,
            root=self.root,
            scene=self.scene,
            split=Split.TRAIN,
        )

        self.test_data = ShanghaiTechDataset(
            task=self.task,
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
            target_frame=self.target_frame,
            root=self.root,
            scene=self.scene,
            split=Split.TEST,
        )

        # Shanghai Tech dataset does not provide a validation set.
        # Auto behaviour is to clone the test set as the validation set.
        if self.val_split_mode == SplitMode.AUTO:
            self.val_data = self.test_data.clone()

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
        """Re-code the training videos to ensure correct reading of frames by torchvision.

        The encoding of the raw video files in the ShanghaiTech dataset causes some problems when
        reading the frames using pyav. To prevent this, we read the frames from the video files using opencv,
        and write them to a new video file that can be parsed correctly with pyav.

        Args:
            video_folder (Path): Path to the folder of training videos.
            target_folder (Path): File system location where the converted videos will be stored.
        """
        training_videos = sorted(video_folder.glob("*"))
        for video_idx, video_path in enumerate(training_videos):
            logger.info("Converting training video %s (%i/%i)...", video_path.name, video_idx + 1, len(training_videos))
            file_name = video_path.name
            target_path = target_folder / file_name
            convert_video(video_path, target_path, codec="XVID")
