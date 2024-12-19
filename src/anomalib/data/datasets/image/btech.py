"""BTech Dataset.

This module provides PyTorch Dataset implementation for the BTech dataset. The
dataset will be downloaded and extracted automatically if not found locally.

The dataset contains 3 categories of industrial objects with both normal and
anomalous samples. Each category includes RGB images and pixel-level ground truth
masks for anomaly segmentation.

License:
    BTech dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    Mishra, P., Verk, C., Fornasier, D., & Piciarelli, C. (2021). VT-ADL: A
    Vision Transformer Network for Image Anomaly Detection and Localization. In
    IEEE International Conference on Image Processing (ICIP), 2021.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
from pandas.core.frame import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.utils import LabelName, Split, validate_path

CATEGORIES = ("01", "02", "03")


class BTechDataset(AnomalibDataset):
    """BTech dataset class.

    Dataset class for loading and processing BTech dataset images. Supports both
    classification and segmentation tasks.

    Args:
        root (Path | str): Path to root directory containing the dataset.
        category (str): Category name, must be one of ``CATEGORIES``.
        transform (Transform | None, optional): Transforms to apply to the images.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split - usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> from pathlib import Path
        >>> from anomalib.data.datasets import BTechDataset
        >>> dataset = BTechDataset(
        ...     root=Path("./datasets/btech"),
        ...     category="01",
        ...     split="train"
        ... )
        >>> dataset[0].keys()
        dict_keys(['image'])

        >>> dataset.split = "test"
        >>> dataset[0].keys()
        dict_keys(['image', 'image_path', 'label'])

        >>> # For segmentation task
        >>> dataset.split = "test"
        >>> dataset[0].keys()
        dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])
        >>> dataset[0]["image"].shape, dataset[0]["mask"].shape
        (torch.Size([3, 256, 256]), torch.Size([256, 256]))
    """

    def __init__(
        self,
        root: str | Path,
        category: str,
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.root_category = Path(root) / category
        self.split = split
        self.samples = make_btech_dataset(path=self.root_category, split=self.split)


def make_btech_dataset(path: Path, split: str | Split | None = None) -> DataFrame:
    """Create BTech samples by parsing the BTech data file structure.

    The files are expected to follow the structure:

    .. code-block:: bash

        path/to/dataset/
        ├── split/
        │   └── category/
        │       └── image_filename.png
        └── ground_truth/
            └── category/
                └── mask_filename.png

    Args:
        path (Path): Path to dataset directory.
        split (str | Split | None, optional): Dataset split - usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> from pathlib import Path
        >>> path = Path("./datasets/btech/01")
        >>> samples = make_btech_dataset(path, split="train")
        >>> samples.head()
           path        split label image_path              mask_path          label_index
        0  BTech/01   train ok    BTech/01/train/ok/105.bmp BTech/01/gt/ok/105.png  0
        1  BTech/01   train ok    BTech/01/train/ok/017.bmp BTech/01/gt/ok/017.png  0

    Returns:
        DataFrame: DataFrame containing samples for the requested split.

    Raises:
        RuntimeError: If no images are found in the dataset directory.
    """
    path = validate_path(path)

    samples_list = [
        (str(path),) + filename.parts[-3:] for filename in path.glob("**/*") if filename.suffix in {".bmp", ".png"}
    ]
    if not samples_list:
        msg = f"Found 0 images in {path}"
        raise RuntimeError(msg)

    samples = pd.DataFrame(samples_list, columns=["path", "split", "label", "image_path"])
    samples = samples[samples.split != "ground_truth"]

    # Create mask_path column
    # (safely handles cases where non-mask image_paths end with either .png or .bmp)
    samples["mask_path"] = (
        samples.path
        + "/ground_truth/"
        + samples.label
        + "/"
        + samples.image_path.str.rstrip("png").str.rstrip(".").str.rstrip("bmp").str.rstrip(".")
        + ".png"
    )

    # Modify image_path column by converting to absolute path
    samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    # Good images don't have mask
    samples.loc[(samples.split == "test") & (samples.label == "ok"), "mask_path"] = ""

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "ok"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "ok"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    # infer the task type
    samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    # Get the data frame for the split.
    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples
