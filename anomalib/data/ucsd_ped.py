"""UCSD Pedestrian dataset."""

import glob
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import albumentations as A
import cv2
import torch
from pandas import DataFrame
from torch import Tensor

from anomalib.data.base import AnomalibDataModule
from anomalib.data.base.video import VideoAnomalibDataset
from anomalib.data.utils import Split, ValSplitMode, read_image
from anomalib.data.utils.video import ClipsIndexer
from anomalib.pre_processing import PreProcessor


def make_ucsd_dataset(path: Path, split: Optional[Union[Split, str]] = None):
    """Create UCSD Pedestrian dataset by parsing the file structure.

    The files are expected to follow the structure:
        path/to/dataset/category/split/video_id/image_filename.tif
        path/to/dataset/category/split/video_id_gt/mask_filename.bmp

    Args:
        root (Path): Path to dataset
        split (Optional[Union[Split, str]], optional): Dataset split (ie., either train or test). Defaults to None.

    Example:
        The following example shows how to get testing samples from UCSDped2 category:

        >>> root = Path('./UCSDped')
        >>> category = 'UCSDped2'
        >>> path = root / category
        >>> path
        PosixPath('UCSDped/UCSDped2')

        >>> samples = make_ucsd_dataset(path, split='test')
        >>> samples.head()
           root             folder image_path                    mask_path                         split
        0  UCSDped/UCSDped2 Test   UCSDped/UCSDped2/Test/Test001 UCSDped/UCSDped2/Test/Test001_gt  test
        1  UCSDped/UCSDped2 Test   UCSDped/UCSDped2/Test/Test002 UCSDped/UCSDped2/Test/Test002_gt  test
        ...

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)
    """
    folders = [filename for filename in sorted(Path(path).glob("*/*")) if os.path.isdir(filename)]
    folders = [folder for folder in folders if len(list(folder.glob("*.tif"))) > 0]

    samples_list = [(str(path),) + folder.parts[-2:] for folder in folders]
    samples = DataFrame(samples_list, columns=["root", "folder", "image_path"])

    samples.loc[samples.folder == "Test", "mask_path"] = samples.image_path.str.split(".").str[0] + "_gt"
    samples.loc[samples.folder == "Test", "mask_path"] = samples.root + "/" + samples.folder + "/" + samples.mask_path
    samples.loc[samples.folder == "Train", "mask_path"] = ""

    samples["image_path"] = samples.root + "/" + samples.folder + "/" + samples.image_path

    samples.loc[samples.folder == "Train", "split"] = "train"
    samples.loc[samples.folder == "Test", "split"] = "test"

    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples


class UCSDpedClips(ClipsIndexer):
    """Clips class for UCSDped dataset."""

    def get_mask(self, idx) -> Optional[Tensor]:
        """Retrieve the masks from the file system."""

        video_idx, frames_idx = self.get_clip_location(idx)
        mask_folder = self.mask_paths[video_idx]
        if mask_folder == "":  # no gt masks available for this clip
            return None
        frames = self.clips[video_idx][frames_idx]

        mask_frames = sorted(glob.glob(mask_folder + "/*"))
        mask_paths = [mask_frames[idx] for idx in frames.int()]

        masks = torch.stack([Tensor(cv2.imread(mask_path, flags=0)) / 255.0 for mask_path in mask_paths])
        return masks

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


class UCSDpedDataset(VideoAnomalibDataset):
    """UCSDped Dataset class.

    Args:
        task (str): Task type, either 'classification' or 'segmentation'
        root (str): Path to the root of the dataset
        category (str): Sub-category of the dataset, e.g. 'bottle'
        pre_process (PreProcessor): Pre-processor object
        split (Optional[Union[Split, str]]): Split of the dataset, usually Split.TRAIN or Split.TEST
        frames_per_clip (int, optional): Number of video frames in each clip.
        stride (int, optional): Number of frames between each consecutive video clip.
    """

    def __init__(
        self,
        task: str,
        root: Union[Path, str],
        category: str,
        pre_process: PreProcessor,
        split: Split,
        frames_per_clip: int = 1,
        stride: int = 1,
    ):
        super().__init__(task, pre_process, frames_per_clip, stride)

        self.root_category = Path(root) / category
        self.split = split
        self.clips_type: Callable = UCSDpedClips

    def _setup(self):
        """Create and assign samples."""
        self.samples = make_ucsd_dataset(self.root_category, self.split)


class UCSDped(AnomalibDataModule):
    """UCSDped DataModule class.

    Args:
        root (str): Path to the root of the dataset
        category (str): Sub-category of the dataset, e.g. 'bottle'
        frames_per_clip (int, optional): Number of video frames in each clip.
        stride (int, optional): Number of frames between each consecutive video clip.
        task (str): Task type, either 'classification' or 'segmentation'
        image_size (Optional[Union[int, Tuple[int, int]]], optional): Size of the input image.
            Defaults to None.
        train_batch_size (int, optional): Training batch size. Defaults to 32.
        eval_batch_size (int, optional): Test batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 8.
        transform_config_train (Optional[Union[str, A.Compose]], optional): Config for pre-processing
            during training.
            Defaults to None.
        transform_config_val (Optional[Union[str, A.Compose]], optional): Config for pre-processing
            during validation.
            Defaults to None.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
    """

    def __init__(
        self,
        root: str,
        category: str,
        frames_per_clip: int = 1,
        stride: int = 1,
        task: str = "segmentation",
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        transform_config_train: Optional[Union[str, A.Compose]] = None,
        transform_config_eval: Optional[Union[str, A.Compose]] = None,
        val_split_mode: ValSplitMode = ValSplitMode.FROM_TEST,
    ):
        super().__init__(train_batch_size, eval_batch_size, num_workers, val_split_mode)

        pre_process_train = PreProcessor(config=transform_config_train, image_size=image_size)
        pre_process_eval = PreProcessor(config=transform_config_eval, image_size=image_size)

        self.train_data = UCSDpedDataset(
            task=task,
            pre_process=pre_process_train,
            frames_per_clip=frames_per_clip,
            stride=stride,
            root=root,
            category=category,
            split=Split.TRAIN,
        )

        self.test_data = UCSDpedDataset(
            task=task,
            pre_process=pre_process_eval,
            frames_per_clip=frames_per_clip,
            stride=stride,
            root=root,
            category=category,
            split=Split.TEST,
        )
