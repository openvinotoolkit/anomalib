"""BTech Dataset.

This script contains PyTorch Dataset for the BTech dataset.

If the dataset is not on the file system, the script downloads and
extracts the dataset and create PyTorch data objects.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
from pandas.core.frame import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.utils import LabelName, Split, validate_path

CATEGORIES = ("01", "02", "03")


class BTechDataset(AnomalibDataset):
    """Btech Dataset class.

    Args:
        root: Path to the BTech dataset
        category: Name of the BTech category.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        split: 'train', 'val' or 'test'
        task: ``classification``, ``detection`` or ``segmentation``
        create_validation_set: Create a validation subset in addition to the train and test subsets

    Examples:
        >>> from anomalib.data.image.btech import BTechDataset
        >>> from anomalib.data.utils.transforms import get_transforms
        >>> transform = get_transforms(image_size=256)
        >>> dataset = BTechDataset(
        ...     task="classification",
        ...     transform=transform,
        ...     root='./datasets/BTech',
        ...     category='01',
        ... )
        >>> dataset[0].keys()
        >>> dataset.setup()
        dict_keys(['image'])

        >>> dataset.split = "test"
        >>> dataset[0].keys()
        dict_keys(['image', 'image_path', 'label'])

        >>> dataset.task = "segmentation"
        >>> dataset.split = "train"
        >>> dataset[0].keys()
        dict_keys(['image'])

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
        transform: Transform | None = None,
        split: str | Split | None = None,
        task: TaskType | str = TaskType.SEGMENTATION,
    ) -> None:
        super().__init__(task, transform)

        self.root_category = Path(root) / category
        self.split = split
        self.samples = make_btech_dataset(path=self.root_category, split=self.split)


def make_btech_dataset(path: Path, split: str | Split | None = None) -> DataFrame:
    """Create BTech samples by parsing the BTech data file structure.

    The files are expected to follow the structure:

        .. code-block:: bash

            path/to/dataset/split/category/image_filename.png
            path/to/dataset/ground_truth/category/mask_filename.png

    Args:
        path (Path): Path to dataset
        split (str | Split | None, optional): Dataset split (ie., either train or test).
            Defaults to ``None``.

    Example:
        The following example shows how to get training samples from BTech 01 category:

        .. code-block:: python

            >>> root = Path('./BTech')
            >>> category = '01'
            >>> path = root / category
            >>> path
            PosixPath('BTech/01')

            >>> samples = make_btech_dataset(path, split='train')
            >>> samples.head()
            path     split label image_path                  mask_path                     label_index
            0  BTech/01 train 01    BTech/01/train/ok/105.bmp BTech/01/ground_truth/ok/105.png      0
            1  BTech/01 train 01    BTech/01/train/ok/017.bmp BTech/01/ground_truth/ok/017.png      0
            ...

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)
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

    # Get the data frame for the split.
    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples
