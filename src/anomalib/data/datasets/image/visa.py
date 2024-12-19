"""Visual Anomaly (VisA) Dataset.

Description:
    This script contains PyTorch Dataset for the Visual Anomal
    (VisA) dataset. If the dataset is not on the file system, the script
    downloads and extracts the dataset and create PyTorch data objects.

License:
    The VisA dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).

Reference:
    - Zou, Y., Jeong, J., Pemula, L., Zhang, D., & Dabeer, O. (2022). SPot-the-Difference
      Self-supervised Pre-training for Anomaly Detection and Segmentation. In European
      Conference on Computer Vision (pp. 392-408). Springer, Cham.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datasets import AnomalibDataset
from anomalib.data.datasets.image.mvtec import make_mvtec_dataset
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

    Args:
        root (str | Path): Path to the root of the dataset
        category (str): Sub-category of the dataset, e.g. 'candle'
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST
            Defaults to ``None``.

    Examples:
        To create a Visa dataset for classification:

        .. code-block:: python

            from anomalib.data.image.visa import VisaDataset
            from anomalib.data.utils.transforms import get_transforms

            transform = get_transforms(image_size=256)
            dataset = VisaDataset(
                transform=transform,
                split="train",
                root="./datasets/visa/visa_pytorch/",
                category="candle",
            )
            dataset.setup()
            dataset[0].keys()

            # Output
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
        self.samples = make_mvtec_dataset(self.root_category, split=self.split, extensions=EXTENSIONS)
