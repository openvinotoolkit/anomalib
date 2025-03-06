"""Visual Anomaly (VisA) Dataset.

This module provides PyTorch Dataset implementation for the Visual Anomaly (VisA)
dataset. The dataset will be downloaded and extracted automatically if not found
locally.

The dataset contains 12 categories of industrial objects with both normal and
anomalous samples. Each category includes RGB images and pixel-level ground truth
masks for anomaly segmentation.

License:
    The VisA dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    Zou, Y., Jeong, J., Pemula, L., Zhang, D., & Dabeer, O. (2022).
    SPot-the-Difference Self-supervised Pre-training for Anomaly Detection and
    Segmentation. In European Conference on Computer Vision (pp. 392-408).
    Springer, Cham.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datasets import AnomalibDataset
from anomalib.data.datasets.image.mvtecad import make_mvtec_ad_dataset
from anomalib.data.utils import Split

EXTENSIONS = (".png", ".jpg", ".JPG")
CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)


class VisaDataset(AnomalibDataset):
    """VisA dataset class.

    Dataset class for loading and processing Visual Anomaly (VisA) dataset images.
    Supports both classification and segmentation tasks.

    Args:
        root (str | Path): Path to root directory containing the dataset.
        category (str): Category name, must be one of ``CATEGORIES``.
        transform (Transform | None, optional): Transforms to apply to the images.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split - usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> from pathlib import Path
        >>> from anomalib.data.datasets import VisaDataset
        >>> dataset = VisaDataset(
        ...     root=Path("./datasets/visa"),
        ...     category="candle",
        ...     split="train"
        ... )
        >>> item = dataset[0]
        >>> item.keys()
        dict_keys(['image_path', 'label', 'image', 'mask'])
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
        self.samples = make_mvtec_ad_dataset(
            self.root_category,
            split=self.split,
            extensions=EXTENSIONS,
        )
