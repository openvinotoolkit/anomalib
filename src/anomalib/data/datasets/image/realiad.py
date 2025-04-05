"""Real-IAD Dataset.

This module provides PyTorch Dataset implementation for the Real-IAD dataset.
The dataset contains 30 categories of industrial objects with both normal and
anomalous samples, captured from 5 different camera viewpoints.

Dataset Structure:
    The dataset follows this directory structure:
        Real-IAD/
        ├── realiad_256/      # 256x256 resolution images
        │   └── CATEGORY/     # e.g. audiojack, button_battery, etc.
        │       ├── OK/       # Normal samples
        │       │   └── SXXXX/  # Sample ID
        │       │       └── CATEGORY_XXXX_OK_CX_TIMESTAMP.jpg
        │       └── NG/       # Anomalous samples
        │           └── DEFECT_TYPE/  # Type of defect
        │               └── SXXXX/
        │                   ├── CATEGORY_XXXX_NG_CX_TIMESTAMP.jpg
        │                   └── CATEGORY_XXXX_NG_CX_TIMESTAMP_mask.png
        ├── realiad_512/      # 512x512 resolution images
        ├── realiad_1024/     # 1024x1024 resolution images
        └── realiad_jsons/    # JSON metadata files
            ├── realiad_jsons/       # Base metadata (multi-view)
            ├── realiad_jsons_sv/    # Single-view metadata
            └── realiad_jsons_fuiad/ # FUIAD metadata versions

License:
    Real-IAD is released under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Sequence
from pathlib import Path

from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.utils import LabelName, Split, validate_path

IMG_EXTENSIONS = (".jpg", ".png", ".PNG", ".JPG")
RESOLUTIONS = ("256", "512", "1024", "raw")
CATEGORIES = (
    "audiojack",
    "button_battery",
    "capacitor",
    "connector",
    "diode",
    "end_cap",
    "fuse",
    "ic",
    "inductor",
    "led",
    "pcb_finger",
    "plastic_nut",
    "potentiometer",
    "relay",
    "resistor",
    "rivet",
    "rubber_grommet",
    "screw",
    "spring",
    "switch",
    "terminal_block",
    "through_hole",
    "toggle_switch",
    "toy_brick",
    "transistor",
    "washer",
    "woodstick",
    "zipper",
    "toothbrush",
    "usb_adaptor",
)


class RealIADDataset(AnomalibDataset):
    """Real-IAD dataset class.

    Dataset class for loading and processing Real-IAD dataset images. Supports
    both classification and segmentation tasks, with multi-view capabilities.

    The dataset provides:
    - 30 industrial object categories
    - 5 camera viewpoints per object (C1-C5)
    - Multiple image resolutions (256x256, 512x512, 1024x1024)
    - Segmentation masks for anomalous samples
    - JSON metadata for flexible dataset organization

    Args:
        root (Path | str): Path to root directory containing the dataset.
            Defaults to ``"./datasets/Real-IAD"``.
        category (str): Category name, must be one of ``CATEGORIES``.
            Defaults to ``"audiojack"``.
        resolution (str | int): Image resolution, must be one of ``RESOLUTIONS`` or their integer equivalents.
            For example, both "256" and 256 are valid. Defaults to ``256``.
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split - usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.
        json_path (str | Path): Path to JSON metadata file, relative to root directory.
            Can use {category} placeholder which will be replaced with the category name.
            Common paths are:
            - "realiad_jsons/realiad_jsons/{category}.json" - Base metadata (multi-view)
            - "realiad_jsons/realiad_jsons_sv/{category}.json" - Single-view metadata
            - "realiad_jsons/realiad_jsons_fuiad_0.4/{category}.json" - FUIAD v0.4 metadata

    Example:
        >>> from pathlib import Path
        >>> from anomalib.data.datasets import RealIADDataset

        >>> # Using base JSON metadata (multi-view) with string resolution
        >>> dataset = RealIADDataset(
        ...     root=Path("./datasets/Real-IAD"),
        ...     category="audiojack",
        ...     resolution="1024",
        ...     split="train",
        ...     json_path="realiad_jsons/realiad_jsons/audiojack.json"
        ... )

        >>> # Using integer resolution
        >>> dataset = RealIADDataset(
        ...     category="button_battery",
        ...     resolution=512
        ... )

        >>> # Using single-view metadata
        >>> dataset = RealIADDataset(
        ...     json_path="realiad_jsons/realiad_jsons_sv/audiojack.json"
        ... )

        >>> # Using FUIAD v0.4 metadata (filtered subset)
        >>> dataset = RealIADDataset(
        ...     json_path="realiad_jsons/realiad_jsons_fuiad_0.4/audiojack.json"
        ... )

        >>> # Using custom JSON file
        >>> dataset = RealIADDataset(
        ...     json_path="path/to/custom/metadata.json"
        ... )

    Notes:
        - Normal samples are in the 'OK' directory, anomalous in 'NG'
        - Each sample has a unique ID (SXXXX) and camera view (CX)
        - Anomalous samples include defect type and segmentation masks
        - JSON metadata provides flexible dataset organization
        - The task (classification/segmentation) is determined by mask availability
    """

    def __init__(
        self,
        root: Path | str = "./datasets/Real-IAD",
        category: str = "audiojack",
        resolution: str | int = 256,
        augmentations: Transform | None = None,
        split: str | Split | None = None,
        json_path: str | Path = "realiad_jsons/realiad_jsons/{category}.json",
    ) -> None:
        """Initialize RealIAD dataset.

        Args:
            root: Path to root directory containing the dataset.
            category: Category name, must be one of ``CATEGORIES``.
            resolution: Image resolution, must be one of ``RESOLUTIONS`` or their integer equivalents.
                For example, both "256" and 256 are valid.
            augmentations: Augmentations that should be applied to the input images.
            split: Dataset split - usually ``Split.TRAIN`` or ``Split.TEST``.
            json_path: Path to JSON metadata file, relative to root directory.
                Can use {category} placeholder which will be replaced with the category name.
                Common paths are:
                - "realiad_jsons/realiad_jsons/{category}.json" - Base metadata (multi-view)
                - "realiad_jsons/realiad_jsons_sv/{category}.json" - Single-view metadata
                - "realiad_jsons/realiad_jsons_fuiad_0.4/{category}.json" - FUIAD v0.4 metadata
        """
        super().__init__(augmentations=augmentations)

        if category not in CATEGORIES:
            msg = f"Category {category} not found in Real-IAD dataset. Available categories: {CATEGORIES}"
            raise ValueError(msg)

        # Convert resolution to string if it's an integer
        if isinstance(resolution, int):
            resolution = str(resolution)

        if resolution not in RESOLUTIONS:
            msg = f"Resolution {resolution} not found in Real-IAD dataset. Available resolutions: {RESOLUTIONS}"
            raise ValueError(msg)

        self.root = Path(root)
        self.category = category
        self.resolution = resolution
        self.split = split

        # Format json_path if it contains {category} placeholder
        if isinstance(json_path, str):
            json_path = json_path.format(category=category)

        # Resolve JSON path
        json_file = self.root / json_path

        # Load JSON metadata
        if not json_file.exists():
            msg = f"JSON metadata file not found at {json_file}"
            raise FileNotFoundError(msg)

        with json_file.open(encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Validate JSON structure
        if not isinstance(self.metadata, dict) or not any(key in self.metadata for key in ["train", "test"]):
            msg = f"Invalid JSON structure in {json_file}. Must contain 'train' and/or 'test' keys."
            raise ValueError(msg)

        # Construct the path to the category directory based on resolution
        self.root_category = self.root / f"realiad_{resolution}" / category

        # Create dataset samples
        self.samples = make_realiad_dataset(
            self.root_category,
            split=self.split,
            extensions=IMG_EXTENSIONS,
            metadata=self.metadata,
        )


def make_realiad_dataset(
    root: str | Path,
    split: str | Split | None = None,
    extensions: Sequence[str] | None = None,
    metadata: dict | None = None,
) -> DataFrame:
    """Create Real-IAD samples by parsing the JSON metadata.

    Args:
        root (Path | str): Path to dataset root directory
        split (str | Split | None, optional): Dataset split (train or test)
            Defaults to ``None``.
        extensions (Sequence[str] | None, optional): Valid file extensions
            Defaults to ``None``.
        metadata (dict | None, optional): JSON metadata containing dataset organization.
            Defaults to ``None``.

    Returns:
        DataFrame: Dataset samples with columns:
            - image_path: Path to image file
            - mask_path: Path to mask file (if available)
            - label_index: Numeric label (0=normal, 1=abnormal)
            - split: Dataset split (train/test)
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = validate_path(root)

    if metadata is None:
        msg = "JSON metadata is required for RealIAD dataset"
        raise ValueError(msg)

    samples_list = []

    # Use train/test splits from JSON metadata
    if split is not None:
        split_key = "train" if split == Split.TRAIN else "test"
        if split_key not in metadata:
            msg = f"Split {split_key} not found in JSON metadata"
            raise ValueError(msg)
        samples = metadata[split_key]
    else:
        # If no split specified, use all samples
        samples = metadata.get("train", []) + metadata.get("test", [])

    for sample in samples:
        # Create sample data with only essential columns
        sample_data = {
            "image_path": str(root / sample["image_path"]),
            "mask_path": str(root / sample["mask_path"]) if sample.get("mask_path") else "",
            "label_index": LabelName.NORMAL if sample["anomaly_class"] == "OK" else LabelName.ABNORMAL,
            "split": "train" if sample in metadata.get("train", []) else "test",
        }
        samples_list.append(sample_data)

    samples = DataFrame(samples_list)

    # Set task type
    samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    return samples
