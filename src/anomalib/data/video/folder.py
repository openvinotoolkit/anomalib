"""Custom video Folder Dataset.

This script creates a custom dataset from a folder.
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import (
    DirType,
    LabelName,
    Split,
    TestSplitMode,
    ValSplitMode,
    validate_path,
)
from anomalib.data.utils.path import _prepare_files_labels, validate_and_resolve_path


def make_folder_video_dataset(
    path: Path | None = None,
    gt_dir: Path | None = None,
    split: str | Split | None = None,
) -> DataFrame:
    """Make Folder Video Dataset.

    Args:
        path (Path | None): Path to the root directory of the dataset.
            Defaults to ``None``.
        gt_dir (Path | None, optional): Path to the directory containing the mask annotations.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split (ie., Split.FULL, Split.TRAIN or Split.TEST).
            Defaults to ``None``.
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the directory.
            Defaults to ``None``.

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test).

    Example:
        The following example shows how to get testing samples from ShanghaiTech dataset:

        >>> root = Path('./shanghaiTech')
        >>> scene = 1
        >>> samples = make_avenue_dataset(path, scene, split='test')
        >>> samples.head()
            root            image_path                          split   mask_path
        0	shanghaitech	shanghaitech/testing/frames/01_0014	test	shanghaitech/testing/test_pixel_mask/01_0014.npy
        1	shanghaitech	shanghaitech/testing/frames/01_0015	test	shanghaitech/testing/test_pixel_mask/01_0015.npy
        ...

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)
    """
    # get paths to training videos
    path = validate_path(path)
    # TODO(bepitic): reflect on the true path in example and doc
    train_path = path / "training"
    train_list = [(str(train_path),) + filename.parts[-2:] for filename in train_path.glob("*.avi")]
    train_samples = DataFrame(train_list, columns=["root", "folder", "image_path"])
    train_samples["split"] = "train"

    # get paths to testing folders
    test_path = Path(path) / "testing"
    test_list = [(str(test_path),) + filename.parts[-2:] for filename in test_path.glob("*.avi")]
    test_samples = DataFrame(
        test_list,
        columns=["root", "folder", "image_path"],
    )  # TODO(Bepitic): Change image Path to video path
    test_samples["split"] = "test"

    samples = pd.concat([train_samples, test_samples], ignore_index=True)

    gt_root = Path(gt_dir) / "testing"
    samples["mask_path"] = ""
    # TODO(Bepitic): Maybe other formats?
    samples.loc[samples.root == str(test_path), "mask_path"] = (
        str(gt_root) + "/" + samples.image_path.str.split(".").str[0] + ".npy"
    )

    samples["image_path"] = samples.root + "/" + samples.image_path

    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples
