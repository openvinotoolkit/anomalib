# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Kaputt Dataset.

This module provides PyTorch Dataset implementation for the Kaputt dataset.

The Kaputt dataset is a large-scale dataset for visual defect detection in
logistics settings. With over 230,000 images (and more than 29,000 defective
instances), it is 40 times larger than MVTec AD and contains more than 48,000
distinct objects.

The dataset uses Parquet files for metadata and has the following structure:
    datasets/
    ├── query-train.parquet
    ├── query-validation.parquet
    ├── query-test.parquet
    ├── reference-train.parquet
    ├── reference-validation.parquet
    └── reference-test.parquet

    query-image/data/<split>/query-data/image/      # Query images (may have defects)
    query-crop/data/<split>/query-data/crop/        # Cropped item regions
    query-mask/data/<split>/query-data/mask/        # Binary segmentation masks

    reference-image/data/<split>/reference-data/image/   # Reference (defect-free) images
    reference-crop/data/<split>/reference-data/crop/
    reference-mask/data/<split>/reference-data/mask/

License:
    The Kaputt dataset is released under the Creative Commons
    Attribution-NonCommercial-NoDerivatives 4.0 International License
    (CC BY-NC-ND 4.0) https://creativecommons.org/licenses/by-nc-nd/4.0/

Reference:
    Höfer, S., Henning, D., Amiranashvili, A., Morrison, D., Tzes, M.,
    Posner, I., Matvienko, M., Rennola, A., & Milan, A. (2025).
    Kaputt: A Large-Scale Dataset for Visual Defect Detection.
    In IEEE/CVF International Conference on Computer Vision (ICCV).

Dataset URL:
    https://huggingface.co/datasets/amazon/kaputt
"""

import logging
from enum import Enum
from functools import partial
from pathlib import Path

import pandas as pd
from lightning_utilities.core.imports import module_available
from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.utils import LabelName, Split, validate_path
from anomalib.data.utils.path import resolve_path_under_root
from anomalib.utils.path import get_datasets_dir

logger = logging.getLogger(__name__)


def _resolve_relative_path(root: Path, subdir: Path, value: str) -> Path:
    """Resolve ``value`` relative to ``subdir`` and confine the result to ``root``.

    Args:
        root: Top-level dataset root that the resolved path must remain under.
        subdir: Subdirectory that ``value`` is relative to.
        value: Relative path read from the dataset metadata.

    Returns:
        Validated absolute path under ``root``.
    """
    return resolve_path_under_root(root, subdir / value, should_exist=False)


# Material categories in Kaputt dataset (based on item_material field)
CATEGORIES = (
    "book_other",
    "book_paper",
    "book_plastic_tight_wrap",
    "cardboard",
    "other",
    "paper",
    "plastic_bubble_wrap",
    "plastic_hard",
    "plastic_loose_bag",
    "plastic_tight_wrap",
)


class ImageType(str, Enum):
    """Type of images to use from the Kaputt dataset.

    Attributes:
        IMAGE: Use full images.
        CROP: Use cropped item regions.
    """

    IMAGE = "image"
    CROP = "crop"


class ImageMode(str, Enum):
    """Controls which image sources are used for training.

    Attributes:
        QUERY_ONLY: Use only query images (default). Query images may
            include both normal and defective samples.
        QUERY_AND_REFERENCE: Use both query and reference images.
            Reference images are always labeled as normal.
        REFERENCE_ONLY: Use only reference (defect-free) images.
            Useful for building a memory bank from reference images only
            (e.g., PatchCore reference-only setting from the Kaputt paper).
    """

    QUERY_ONLY = "query_only"
    QUERY_AND_REFERENCE = "query_and_reference"
    REFERENCE_ONLY = "reference_only"


class KaputtDataset(AnomalibDataset):
    """Kaputt dataset class.

    Dataset class for loading and processing Kaputt dataset images. Supports
    both classification and segmentation tasks.

    The Kaputt dataset uses Parquet files for metadata instead of a directory
    structure. Images are organized as query (potentially defective) and
    reference (defect-free) sets.

    Args:
        root (Path | str | None): Path to root directory containing the dataset.
            Defaults to ``None``.
        category (str | None): Category of the dataset (maps to ``item_material``).
            Defaults to ``None`` which loads all categories.
        augmentations (Transform, optional): Augmentations that should be applied
            to the input images. Defaults to ``None``.
        split (str | Split | None, optional): Dataset split - ``Split.TRAIN``,
            ``Split.VAL``, or ``Split.TEST``. Defaults to ``None``.
        image_type (ImageType | str): Type of images to use.
            Defaults to ``ImageType.IMAGE``.
        image_mode (ImageMode | str): Controls which image sources are used.
            Defaults to ``ImageMode.QUERY_ONLY``.

    Example:
        >>> from pathlib import Path
        >>> from anomalib.data.datasets import KaputtDataset
        >>> dataset = KaputtDataset(
        ...     root=Path("./datasets/kaputt"),
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

        Images are PyTorch tensors with shape ``(C, H, W)``, masks have shape
        ``(H, W)``:

        >>> sample["image"].shape, sample["mask"].shape
        (torch.Size([3, 256, 256]), torch.Size([256, 256]))
    """

    def __init__(
        self,
        root: Path | str | None = None,
        category: str | None = None,
        augmentations: Transform | None = None,
        split: str | Split | None = None,
        image_type: ImageType | str = ImageType.IMAGE,
        image_mode: ImageMode | str = ImageMode.QUERY_ONLY,
    ) -> None:
        super().__init__(augmentations=augmentations)

        root = root if root is not None else get_datasets_dir() / "kaputt"

        self.root = Path(root)
        self.category: str | None = category
        self.split = split
        self.image_type = ImageType(image_type)
        self.image_mode = ImageMode(image_mode)
        self.samples = make_kaputt_dataset(
            self.root,
            category=self.category,
            split=self.split,
            image_type=self.image_type,
            image_mode=self.image_mode,
        )


def make_kaputt_dataset(
    root: str | Path,
    category: str | None = None,
    split: str | Split | None = None,
    image_type: ImageType | str = ImageType.IMAGE,
    image_mode: ImageMode | str = ImageMode.QUERY_ONLY,
) -> DataFrame:
    """Create Kaputt samples by parsing the Parquet metadata files.

    The Kaputt dataset uses Parquet files containing metadata about each image,
    including relative paths to image, crop, and mask files. Query parquets
    contain columns such as ``capture_id``, ``defect``, ``item_material``,
    ``query_image``, ``query_crop``, and ``query_mask``. Reference parquets
    contain ``item_identifier``, ``reference_image``, ``reference_crop``, and
    ``reference_mask``.

    Args:
        root (Path | str): Path to dataset root directory.
        category (str | None): Category of the dataset (maps to
            ``item_material``). Defaults to ``None`` which loads all categories.
        split (str | Split | None, optional): Dataset split (train, val, or test).
            Defaults to ``None`` which loads all splits.
        image_type (ImageType | str): Type of images to use.
            Defaults to ``ImageType.IMAGE``.
        image_mode (ImageMode | str): Controls which image sources are used.
            Defaults to ``ImageMode.QUERY_ONLY``.

    Returns:
        DataFrame: Dataset samples with columns:
            - split: Dataset split (train/val/test)
            - label: Class label ("normal" or "abnormal")
            - image_path: Path to image file
            - mask_path: Path to mask file (for abnormal images)
            - label_index: Numeric label (0=normal, 1=abnormal)
            - capture_id: Unique capture identifier
            - item_identifier: Item identifier linking query to reference images
            - defect_types: List of defect types (for abnormal images)
            - item_material: Material category

    Example:
        >>> root = Path("./datasets/kaputt")
        >>> samples = make_kaputt_dataset(root, split="train")
        >>> samples.head()
           split label image_path           mask_path label_index capture_id
        0  train normal [...]/image/abc.jpg                    0      abc

    Raises:
        FileNotFoundError: If the Parquet metadata files are not found.
        RuntimeError: If no valid images are found.
    """
    root = validate_path(root)
    image_type = ImageType(image_type)
    image_mode = ImageMode(image_mode)

    # Parquet files use "validation" while anomalib uses "val"
    parquet_splits = {"train": "train", "validation": "val", "test": "test"}

    frames: list[DataFrame] = []
    include_query = image_mode != ImageMode.REFERENCE_ONLY
    include_reference = image_mode != ImageMode.QUERY_ONLY

    if not module_available("pyarrow"):
        msg = (
            "pyarrow is needed to read the parquet files. You can install it using: `uv pip install pyarrow`"
            "or `uv pip install anomalib[datasets]`"
        )
        raise ImportError(msg)

    for parquet_name, anomalib_name in parquet_splits.items():
        # --- Query samples ---
        if include_query:
            query_parquet = root / "datasets" / f"query-{parquet_name}.parquet"
            if not query_parquet.exists():
                msg = f"Query parquet file not found: {query_parquet}"
                raise FileNotFoundError(msg)

            query_df = pd.read_parquet(query_parquet)

            image_col = f"query_{image_type.value}"  # "query_image" or "query_crop"
            image_subdir = root / f"query-{image_type.value}"
            mask_subdir = root / "query-mask"
            resolve_image = partial(_resolve_relative_path, root, image_subdir)
            resolve_mask = partial(_resolve_relative_path, root, mask_subdir)

            samples = DataFrame()
            samples["image_path"] = query_df[image_col].map(resolve_image).astype(str)
            samples["capture_id"] = query_df["capture_id"]
            samples["item_identifier"] = query_df["item_identifier"] if "item_identifier" in query_df.columns else ""
            samples["item_material"] = query_df["item_material"].fillna("")
            samples["defect_types"] = query_df["defect_types"].fillna("")
            samples["defect"] = query_df["defect"]
            samples["query_mask"] = query_df["query_mask"]
            samples["split"] = anomalib_name

            if category:
                samples = samples[samples["item_material"] == category]

            if samples.empty:
                msg = f"No samples found for category {category} in {query_parquet}"
                logger.warning(msg)
                continue

            samples.loc[samples["defect"] == False, "label_index"] = LabelName.NORMAL  # noqa: E712
            samples.loc[samples["defect"] != False, "label_index"] = LabelName.ABNORMAL  # noqa: E712
            samples["label_index"] = samples["label_index"].astype(int)
            samples.loc[samples["label_index"] == LabelName.NORMAL, "label"] = "normal"
            samples.loc[samples["label_index"] == LabelName.ABNORMAL, "label"] = "abnormal"

            samples["mask_path"] = ""
            abnormal_mask = samples["label_index"] == LabelName.ABNORMAL
            samples.loc[abnormal_mask, "mask_path"] = (
                samples.loc[abnormal_mask, "query_mask"].map(resolve_mask).astype(str)
            )

            samples = samples.drop(columns=["defect", "query_mask"])

            frames.append(samples)

        # --- Reference samples (always normal) ---
        if include_reference:
            ref_parquet = root / "datasets" / f"reference-{parquet_name}.parquet"
            if ref_parquet.exists():
                ref_df = pd.read_parquet(ref_parquet)
                ref_image_col = f"reference_{image_type.value}"
                reference_subdir = root / f"reference-{image_type.value}"
                resolve_ref = partial(_resolve_relative_path, root, reference_subdir)

                ref_samples = DataFrame()
                ref_samples["image_path"] = ref_df[ref_image_col].map(resolve_ref).astype(str)
                ref_samples["capture_id"] = ref_df["item_identifier"]
                ref_samples["item_identifier"] = ref_df["item_identifier"]
                ref_samples["item_material"] = ""
                ref_samples["defect_types"] = ""
                ref_samples["split"] = anomalib_name
                ref_samples["label"] = "normal"
                ref_samples["label_index"] = int(LabelName.NORMAL)
                ref_samples["mask_path"] = ""

                frames.append(ref_samples)
            else:
                msg = f"Reference parquet file not found: {ref_parquet}"
                logger.warning(msg)

    if not frames:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    samples = pd.concat(frames, ignore_index=True)
    samples = samples.sort_values(by="image_path", ignore_index=True)

    # infer the task type
    samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    if split:
        split_value = split.value if isinstance(split, Split) else split
        samples = samples[samples.split == split_value].reset_index(drop=True)

    return samples
