"""Custom Folder Dataset.

This script creates a custom dataset from a folder.
"""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import logging
import tarfile
from distutils import extension
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.request import urlretrieve

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS

from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import DownloadProgressBar, read_image
from anomalib.data.utils.split import (
    create_validation_set_from_test_set,
    split_normal_images_in_train_set,
)
from anomalib.pre_processing import PreProcessor

logger = logging.getLogger(name="Dataset: MVTec")
logger.setLevel(logging.DEBUG)


def __check_and_convert_path(path: Union[str, Path]) -> Path:
    """Check an input path, and convert to Pathlib object.

    Args:
        path (Union[str, Path]): Input path.

    Returns:
        Path: Output path converted to pathlib object.
    """
    if not isinstance(path, Path):
        path = Path(path)
    return path


def make_dataset(
    normal_dir: Path,
    abnormal_dir: Path,
    split: Optional[str] = None,
    split_ratio: float = 0.1,
    seed: int = 0,
    create_validation_set: bool = False,
    extensions: Optional[Tuple[str, ...]] = None,
) -> DataFrame:
    """Create a folder dataset."""

    normal_dir = __check_and_convert_path(normal_dir)
    abnormal_dir = __check_and_convert_path(abnormal_dir)

    if extensions is None:
        extensions = IMG_EXTENSIONS

    normal_filenames = [f for f in normal_dir.glob(r"**/*") if f.suffix in extensions]
    abnormal_filenames = [f for f in abnormal_dir.glob(r"**/*") if f.suffix in extensions]

    # TODO: Create a pd dataframe based on the above filenames.

    # samples_list = [(str(path),) + filename.parts[-3:] for filename in path.glob("**/*.png")]
    # if len(samples_list) == 0:
    #     raise RuntimeError(f"Found 0 images in {path}")

    # samples = pd.DataFrame(samples_list, columns=["path", "split", "label", "image_path"])
    # samples = samples[samples.split != "ground_truth"]

    # # Create mask_path column
    # samples["mask_path"] = (
    #     samples.path
    #     + "/ground_truth/"
    #     + samples.label
    #     + "/"
    #     + samples.image_path.str.rstrip("png").str.rstrip(".")
    #     + "_mask.png"
    # )

    # # Modify image_path column by converting to absolute path
    # samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    # # Split the normal images in training set if test set doesn't
    # # contain any normal images. This is needed because AUC score
    # # cannot be computed based on 1-class
    # if sum((samples.split == "test") & (samples.label == "good")) == 0:
    #     samples = split_normal_images_in_train_set(samples, split_ratio, seed)

    # # Good images don't have mask
    # samples.loc[(samples.split == "test") & (samples.label == "good"), "mask_path"] = ""

    # # Create label index for normal (0) and anomalous (1) images.
    # samples.loc[(samples.label == "good"), "label_index"] = 0
    # samples.loc[(samples.label != "good"), "label_index"] = 1
    # samples.label_index = samples.label_index.astype(int)

    # if create_validation_set:
    #     samples = create_validation_set_from_test_set(samples, seed=seed)

    # # Get the data frame for the split.
    # if split is not None and split in ["train", "val", "test"]:
    #     samples = samples[samples.split == split]
    #     samples = samples.reset_index(drop=True)

    # return samples


class FolderDataset(Dataset):
    """Folder Dataset."""

    def __init__(
        self,
        root: Union[str, Path],
        normal: Union[Path, str],
        abnormal: Union[Path, str],
        split: str,
        mask: Optional[Union[Path, str]] = None,
        pre_process: Optional[PreProcessor] = None,
        extensions: Optional[Sequence[str]] = None,
        task: str = "segmentation",
        seed: int = 0,
        create_validation_set: bool = False,
    ) -> None:
        pass

    def __len__(self) -> int:
        """Get length of the dataset."""
        pass

    def __getitem__(self, index: int) -> Dict[str, Union[str, Tensor]]:
        """Get dataset item for the index ``index``.

        Args:
            index (int): Index to get the item.

        Returns:
            Union[Dict[str, Tensor], Dict[str, Union[str, Tensor]]]: Dict of image tensor during training.
                Otherwise, Dict containing image path, target path, image tensor, label and transformed bounding box.
        """
        pass


samples = make_dataset(
    normal_dir="/home/sakcay/projects/anomalib/datasets/MVTec/bottle/test/good",
    abnormal_dir="/home/sakcay/projects/anomalib/datasets/MVTec/bottle/test/broken_large",
)
