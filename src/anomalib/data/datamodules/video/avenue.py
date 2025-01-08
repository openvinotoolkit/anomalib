"""CUHK Avenue Data Module.

This module provides a PyTorch Lightning DataModule for the CUHK Avenue dataset. If
the dataset is not already present on the file system, the DataModule class will
download and extract the dataset, converting the ``.mat`` mask files to ``.png``
format.

Example:
    Create an Avenue datamodule::

        >>> from anomalib.data import Avenue
        >>> datamodule = Avenue(
        ...     root="./datasets/avenue",
        ...     clip_length_in_frames=2,
        ...     frames_between_clips=1,
        ... )
        >>> datamodule.setup()
        >>> i, data = next(enumerate(datamodule.train_dataloader()))
        >>> data.keys()
        dict_keys(['image', 'video_path', 'frames', 'last_frame', 'original_image'])

Notes:
    The directory structure after preparation will be::

        root/
        ├── ground_truth_demo/
        │   ├── ground_truth_show.m
        │   ├── Readme.txt
        │   ├── testing_label_mask/
        │   └── testing_videos/
        ├── testing_videos/
        │   ├── ...
        │   └── 21.avi
        ├── testing_vol/
        │   ├── ...
        │   └── vol21.mat
        ├── training_videos/
        │   ├── ...
        │   └── 16.avi
        └── training_vol/
            ├── ...
            └── vol16.mat

License:
    The CUHK Avenue dataset is released for academic research only. For licensing
    details, see the original dataset website.

Reference:
    Lu, Cewu, Jianping Shi, and Jiaya Jia. "Abnormal event detection at 150 fps
    in Matlab." In Proceedings of the IEEE International Conference on Computer
    Vision, 2013.
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from pathlib import Path
from shutil import move

import cv2
import scipy.io
from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.video import AnomalibVideoDataModule
from anomalib.data.datasets.base.video import VideoTargetFrame
from anomalib.data.datasets.video.avenue import AvenueDataset
from anomalib.data.utils import DownloadInfo, Split, ValSplitMode, download_and_extract

logger = logging.getLogger(__name__)

DATASET_DOWNLOAD_INFO = DownloadInfo(
    name="Avenue Dataset",
    url="http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/Avenue_Dataset.zip",
    hashsum="fc9cb8432a11ca79c18aa180c72524011411b69d3b0ff27c8816e41c0de61531",
)
ANNOTATIONS_DOWNLOAD_INFO = DownloadInfo(
    name="Avenue Annotations",
    url="http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/ground_truth_demo.zip",
    hashsum="60fec1728ec8f73a58aad3aeb5729d70a805a47e0b8eb4bf91ab67ef06386d77",
)


class Avenue(AnomalibVideoDataModule):
    """Avenue DataModule class.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/avenue"``.
        gt_dir (Path | str): Path to the ground truth files.
            Defaults to ``"./datasets/avenue/ground_truth_demo"``.
        clip_length_in_frames (int): Number of video frames in each clip.
            Defaults to ``2``.
        frames_between_clips (int): Number of frames between consecutive clips.
            Defaults to ``1``.
        target_frame (VideoTargetFrame | str): Target frame in clip for ground
            truth. Defaults to ``VideoTargetFrame.LAST``.
        train_batch_size (int): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int): Test batch size.
            Defaults to ``32``.
        num_workers (int): Number of workers.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply dto the training images
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
        val_split_mode (ValSplitMode | str): How validation subset is obtained.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of data reserved for validation.
            Defaults to ``0.5``.
        seed (int | None): Seed for reproducibility.
            Defaults to ``None``.

    Example:
        Create a dataloader for classification::

            >>> datamodule = Avenue(
            ...     clip_length_in_frames=2,
            ...     frames_between_clips=1,
            ...     target_frame=VideoTargetFrame.LAST
            ... )
            >>> datamodule.setup()
            >>> i, data = next(enumerate(datamodule.train_dataloader()))
            >>> data["image"].shape
            torch.Size([32, 2, 3, 256, 256])

    Notes:
        The dataloader returns batches of clips, where each clip contains
        ``clip_length_in_frames`` consecutive frames. ``frames_between_clips``
        determines frame spacing between clips. ``target_frame`` specifies which
        frame provides ground truth.
    """

    def __init__(
        self,
        root: Path | str = "./datasets/avenue",
        gt_dir: Path | str = "./datasets/avenue/ground_truth_demo",
        clip_length_in_frames: int = 2,
        frames_between_clips: int = 1,
        target_frame: VideoTargetFrame | str = VideoTargetFrame.LAST,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
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
        self.gt_dir = Path(gt_dir)
        self.clip_length_in_frames = clip_length_in_frames
        self.frames_between_clips = frames_between_clips
        self.target_frame = VideoTargetFrame(target_frame)

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = AvenueDataset(
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
            target_frame=self.target_frame,
            root=self.root,
            gt_dir=self.gt_dir,
            split=Split.TRAIN,
        )

        self.test_data = AvenueDataset(
            clip_length_in_frames=self.clip_length_in_frames,
            frames_between_clips=self.frames_between_clips,
            target_frame=self.target_frame,
            root=self.root,
            gt_dir=self.gt_dir,
            split=Split.TEST,
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available.

        This method checks if the specified dataset is available in the file
        system. If not, it downloads and extracts the dataset into the appropriate
        directory.

        Example:
            Assume the dataset is not available on the file system::

                >>> datamodule = Avenue()
                >>> datamodule.prepare_data()

            The directory structure after preparation will be::

                datasets/
                └── avenue/
                    ├── ground_truth_demo/
                    │   ├── ground_truth_show.m
                    │   ├── Readme.txt
                    │   ├── testing_label_mask/
                    │   └── testing_videos/
                    ├── testing_videos/
                    │   ├── ...
                    │   └── 21.avi
                    ├── testing_vol/
                    │   ├── ...
                    │   └── vol21.mat
                    ├── training_videos/
                    │   ├── ...
                    │   └── 16.avi
                    └── training_vol/
                        ├── ...
                        └── vol16.mat
        """
        if self.root.is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DATASET_DOWNLOAD_INFO)
            download_and_extract(self.gt_dir, ANNOTATIONS_DOWNLOAD_INFO)

            # move contents to root
            folder_names = ["Avenue Dataset", "ground_truth_demo"]
            for root, folder_name in zip([self.root, self.gt_dir], folder_names, strict=True):
                extracted_folder = root / folder_name
                for filename in extracted_folder.glob("*"):
                    move(str(filename), str(root / filename.name))
                extracted_folder.rmdir()

            # convert masks
            self._convert_masks(self.gt_dir)

    @staticmethod
    def _convert_masks(gt_dir: Path) -> None:
        """Convert mask files from ``.mat`` to ``.png`` format.

        The masks in the Avenue datasets are provided as matlab (``.mat``) files.
        To speed up data loading, we convert the masks into a separate ``.png``
        file for every video frame in the dataset.

        Args:
            gt_dir (Path): Ground truth folder of the dataset.
        """
        # convert masks to numpy
        masks_dir = gt_dir / "testing_label_mask"
        # get file names
        mat_files = list(masks_dir.glob("*.mat"))
        mask_folders = [matfile.with_suffix("") for matfile in mat_files]
        if not all(folder.exists() for folder in mask_folders):
            # convert mask files to images
            logger.info("converting mat files to .png format.")
            for mat_file, mask_folder in zip(mat_files, mask_folders, strict=True):
                mat = scipy.io.loadmat(mat_file)
                mask_folder.mkdir(parents=True, exist_ok=True)
                masks = mat["volLabel"].squeeze()
                for idx, mask in enumerate(masks):
                    filename = (mask_folder / str(idx).zfill(int(math.log10(len(masks)) + 1))).with_suffix(".png")
                    cv2.imwrite(str(filename), mask)
