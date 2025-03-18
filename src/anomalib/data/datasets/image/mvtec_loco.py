"""MVTec LOCO AD Dataset (CC BY-NC-SA 4.0).

Description:
    This script contains PyTorch Dataset, Dataloader and PyTorch Lightning
    DataModule for the MVTec LOCO AD dataset. If the dataset is not on the file system,
    the script downloads and extracts the dataset and create PyTorch data objects.

License:
    MVTec LOCO AD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).

References:
    - Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, and Carsten Steger:
      Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization;
      in: International Journal of Computer Vision (IJCV) 130, 947-969, 2022, DOI: 10.1007/s11263-022-01578-9
"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from pandas import DataFrame
from PIL import Image as PILImage
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2.functional import to_dtype_image, to_image
from torchvision.tv_tensors import Image, Mask

from anomalib.data.dataclasses.torch import ImageItem
from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import (
    DownloadInfo,
    LabelName,
    Split,
    read_image,
    validate_path,
)

logger = logging.getLogger(__name__)


IMG_EXTENSIONS = (".png", ".PNG")

DOWNLOAD_INFO = DownloadInfo(
    name="mvtec_loco",
    url="https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701"
    "/mvtec_loco_anomaly_detection.tar.xz",
    hashsum="9e7c84dba550fd2e59d8e9e231c929c45ba737b6b6a6d3814100f54d63aae687",
)

CATEGORIES = (
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
)


class MVTecLOCODataset(AnomalibDataset):
    """MVTec LOCO dataset class.

    Dataset class for loading and processing MVTec LOCO AD dataset images. Supports
    classification, detection and segmentation tasks.

    Args:
        root (Path | str): Path to root directory containing the dataset.
            Defaults to ``"./datasets/MVTec_LOCO"``.
        category (str): Category name, must be one of ``CATEGORIES``.
            Defaults to ``"breakfast_box"``.
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split - usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> from anomalib.data.datasets import MVTecLocoDataset
        >>> dataset = MVTecLocoDataset(
        ...     root="./datasets/MVTec_LOCO",
        ...     category="breakfast_box",
        ...     split="train"
        ... )

        For classification tasks, each sample contains:

        >>> sample = dataset[0]
        >>> list(sample.keys())
        ['image_path', 'label', 'image']

        For segmentation tasks, samples also include mask paths and masks:

        >>> dataset.task = "segmentation"
        >>> sample = dataset[0]
        >>> list(sample.keys())
        ['image_path', 'label', 'image', 'mask_path', 'mask']

        For detection tasks, samples include boxes:

        >>> dataset.task = "detection"
        >>> sample = dataset[0]
        >>> list(sample.keys())
        ['image_path', 'label', 'image', 'mask_path', 'mask', 'boxes']

        Images are PyTorch tensors with shape ``(C, H, W)``, masks have shape
        ``(H, W)``:

        >>> sample["image"].shape, sample["mask"].shape
        (torch.Size([3, 256, 256]), torch.Size([256, 256]))
    """

    def __init__(
        self,
        root: Path | str = "./datasets/MVTec_LOCO",
        category: str = "breakfast_box",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.root_category = Path(root) / category
        self.category = category
        self.split = split
        self.samples = make_dataset(
            self.root_category,
            split=self.split,
            extensions=IMG_EXTENSIONS,
        )

    @staticmethod
    def _read_mask(mask_path: str | Path) -> Mask:
        """Read mask from path and convert to Mask tensor.

        Args:
            mask_path (str | Path): Path to mask file

        Returns:
            Mask: Mask tensor of shape [H, W] with dtype torch.bool
        """
        image = PILImage.open(mask_path).convert("L")
        return Mask(to_image(image).squeeze() > 0)

    def __getitem__(self, index: int) -> ImageItem:
        """Get a dataset item.

        Args:
            index (int): Index of the item to get.

        Returns:
            ImageItem: The dataset item.
        """
        image_path = self.samples.iloc[index].image_path
        mask_path = self.samples.iloc[index].mask_path
        label_index = self.samples.iloc[index].label_index

        image = read_image(image_path, as_tensor=True)
        item = {"image_path": image_path, "gt_label": label_index}

        # Some of the categories in MVTec LOCO have multiple masks for the same image.
        if isinstance(mask_path, str):
            mask_path = [mask_path]

        # Only Anomalous (1) images have masks in anomaly datasets
        # Therefore, create empty mask for Normal (0) images.
        semantic_mask = (
            Mask(torch.zeros(image.shape[-2:], dtype=torch.bool))
            if label_index == LabelName.NORMAL
            else Mask(torch.stack([self._read_mask(path) for path in mask_path]))
        )

        binary_mask = Mask(semantic_mask.view(-1, *semantic_mask.shape[-2:]).any(dim=0))
        item["image"], item["gt_mask"] = (
            self.augmentations(image, binary_mask) if self.augmentations else (image, binary_mask)
        )

        item["mask_path"] = mask_path
        # List of masks with the original size for saturation based metrics calculation
        item["semantic_mask"] = semantic_mask

        return ImageItem(
            image=Image(to_dtype_image(image, torch.float32, scale=True)),
            gt_mask=binary_mask,
            gt_label=torch.tensor(label_index),
            image_path=image_path,
            mask_path=mask_path[0] if isinstance(mask_path, list) else mask_path,
        )


def make_dataset(
    root: str | Path,
    split: str | Split | None = None,
    extensions: Sequence[str] = IMG_EXTENSIONS,
) -> DataFrame:
    """Create MVTec LOCO AD samples by parsing the original MVTec LOCO AD data file structure.

    The files are expected to follow the structure:
        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/image_filename/000.png

    where there can be multiple ground-truth masks for the corresponding anomalous images.

    This function creates a dataframe to store the parsed information based on the following format:

    +---+---------------+-------+---------+-------------------------+-----------------------------+-------------+
    |   | path          | split | label   | image_path              | mask_path                  | label_index |
    +===+===============+=======+=========+===============+=======================================+=============+
    | 0 | datasets/name | test  | defect  | path/to/image/file.png  | [path/to/masks/file.png]    | 1           |
    +---+---------------+-------+---------+-------------------------+-----------------------------+-------------+

    Args:
        root (str | Path): Path to dataset
        split (str | Split | None): Dataset split (ie., either train or test).
            Defaults to ``None``.
        extensions (Sequence[str]): List of file extensions to be included in the dataset.
            Defaults to ``None``.

    Returns:
        DataFrame: an output dataframe containing the samples of the dataset.

    Examples:
        The following example shows how to get test samples from MVTec LOCO AD pushpins category:

        >>> root = Path('./MVTec_LOCO')
        >>> category = 'pushpins'
        >>> path = root / category
        >>> samples = make_mvtec_loco_dataset(path, split='test')
        >>> samples.head()
           path                split label image_path           mask_path label_index
        0  datasets/MVTec_LOCO/pushpins test good  [...]/good/105.png           0
        1  datasets/MVTec_LOCO/pushpins test good  [...]/good/017.png           0

    Raises:
        RuntimeError: If no valid images are found
        MisMatchError: If anomalous images and masks don't match
    """
    root = validate_path(root)

    # Retrieve the image and mask files
    samples_list = []
    for f in root.glob("**/*"):
        if f.suffix in extensions:
            parts = f.parts
            # 'ground_truth' and non 'ground_truth' path have a different structure
            if "ground_truth" not in parts:
                split_folder, label_folder, image_file = parts[-3:]
                image_path = f"{root}/{split_folder}/{label_folder}/{image_file}"
                samples_list.append((str(root), split_folder, label_folder, "", image_path))
            else:
                split_folder, label_folder, image_folder, image_file = parts[-4:]
                image_path = f"{root}/{split_folder}/{label_folder}/{image_folder}/{image_file}"
                samples_list.append((str(root), split_folder, label_folder, image_folder, image_path))

    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    samples = DataFrame(samples_list, columns=["path", "split", "label", "image_folder", "image_path"])

    # Replace validation to Split.VAL.value in the split column
    samples["split"] = samples["split"].replace("validation", Split.VAL.value)

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    # separate ground-truth masks from samples
    mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(by="image_path", ignore_index=True)
    samples = samples[samples.split != "ground_truth"].sort_values(by="image_path", ignore_index=True)

    # Group masks and aggregate the path into a list
    mask_samples = (
        mask_samples.groupby(["path", "split", "label", "image_folder"])["image_path"]
        .agg(list)
        .reset_index()
        .rename(columns={"image_path": "mask_path"})
    )

    # assign mask paths to anomalous test images
    samples["mask_path"] = ""
    samples.loc[
        (samples.split == "test") & (samples.label_index == LabelName.ABNORMAL),
        "mask_path",
    ] = mask_samples.mask_path.to_numpy()

    # validate that the right mask files are associated with the right test images
    if len(samples.loc[samples.label_index == LabelName.ABNORMAL]):
        image_stems = samples.loc[samples.label_index == LabelName.ABNORMAL]["image_path"].apply(lambda x: Path(x).stem)
        mask_parent_stems = samples.loc[samples.label_index == LabelName.ABNORMAL]["mask_path"].apply(
            lambda x: {Path(mask_path).parent.stem for mask_path in x},
        )

        if not all(
            next(iter(mask_stems)) == image_stem
            for image_stem, mask_stems in zip(image_stems, mask_parent_stems, strict=True)
        ):
            msg = (
                "Mismatch between anomalous images and ground truth masks. "
                "Make sure the parent folder of the mask files in 'ground_truth' folder "
                "follows the same naming convention as the anomalous images in the dataset "
                "(e.g., image: '005.png', mask: '005/000.png')."
            )
            raise MisMatchError(msg)

    # infer the task type
    samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples
