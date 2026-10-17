# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LIBAD Dataset.

This module provides PyTorch Dataset implementation for the LIBAD dataset.
The dataset will not be downloaded automatically. Users must download it from
Hugging Face (https://huggingface.co/datasets/Evenrose/LIBAD) and extract it.

The dataset contains 11 categories of Li-Ion Battery Electrode Manufacturing
defects. Each sample has multiple modalities (e.g. VIS, X-ray). Modalities are
distinguished by the suffix of the image filename (e.g. A01A.tiff, A01B.tiff,
A01X.tiff, A01L.tiff).

License:
    LIBAD dataset is released under the CC-BY-4.0 License.

Reference:
    Wenbo Sui and Daniel Lichau and Harold Phelippeau and Zhao Liu. (2026).
    LIBAD: A Multimodal Anomaly Detection Benchmark for Li-Ion Battery Electrode Manufacturing.
"""

from pathlib import Path

import pandas as pd
from pandas.core.frame import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.utils import LabelName, Split, validate_path

CATEGORIES = (
    "1_wrinkling",
    "2_particle",
    "3_pit",
    "4_unevenness",
    "5_barefoil",
    "6_scratch",
    "7_polarity",
    "8_debonding",
    "9_crack",
    "10_streak",
    "11_pinhole",
)


class LIBADDataset(AnomalibDataset):
    """LIBAD dataset class.

    Dataset class for loading and processing LIBAD dataset images. Supports
    classification tasks.

    Args:
        root (Path | str): Path to root directory containing the dataset.
        category (str): Category name, must be one of ``CATEGORIES``.
        modality (str, optional): Modality suffix to filter images by.
            Options: 'A' (side A), 'B' (side B), 'L' (X-ray Low), 'X' (X-ray High).
            Defaults to ``'A'``.
        augmentations (Transform | None, optional): Transforms to apply to the images.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split - usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> from pathlib import Path
        >>> from anomalib.data.datasets import LIBADDataset
        >>> dataset = LIBADDataset(
        ...     root=Path("./datasets/LIBAD"),
        ...     category="1_wrinkling",
        ...     modality="A",
        ...     split="train"
        ... )
        >>> dataset[0].keys()
        dict_keys(['image_path', 'label', 'image'])
    """

    def __init__(
        self,
        root: str | Path,
        category: str,
        modality: str = "A",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        if category not in CATEGORIES:
            msg = f"Unknown category {category}. Must be one of {CATEGORIES}"
            raise ValueError(msg)

        if modality is not None and modality not in {"A", "B", "L", "X"}:
            msg = f"Unknown modality {modality}. Must be one of 'A', 'B', 'L', 'X'"
            raise ValueError(msg)

        self.root_category = Path(root) / category
        self.modality = modality
        self.split = split
        self.samples = make_libad_dataset(path=self.root_category, modality=self.modality, split=self.split)


def make_libad_dataset(path: Path, modality: str = "A", split: str | Split | None = None) -> DataFrame:
    """Create LIBAD samples by parsing the LIBAD data file structure.

    The files are expected to follow the structure:

    .. code-block:: bash

        path/to/dataset/
        ├── category/
        │   ├── normal/
        │   │   ├── A01A.tiff
        │   │   ├── A01B.tiff
        │   │   ├── A01X.tiff
        │   │   └── A01L.tiff
        │   └── anomaly/
        │       ├── ...

    Args:
        path (Path): Path to dataset directory.
        modality (str, optional): Modality suffix to filter images by.
            Defaults to ``'A'``.
        split (str | Split | None, optional): Dataset split - usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.

    Returns:
        DataFrame: DataFrame containing samples for the requested split.

    Raises:
        RuntimeError: If no images are found in the dataset directory.
    """
    path = validate_path(path)

    samples_list = [
        (str(path), filename.parent.name, str(filename))
        for filename in path.glob("**/*")
        if filename.suffix in {".tiff", ".tif"} and filename.stem.endswith(modality)
    ]
    if not samples_list:
        msg = f"Found 0 images in {path} with modality {modality}"
        raise RuntimeError(msg)

    samples = pd.DataFrame(samples_list, columns=["path", "label", "image_path"])

    # Set split to None by default so the datamodule handles train/test splitting
    samples["split"] = None

    # Create mask_path column (LIBAD does not have masks)
    samples["mask_path"] = ""

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "normal"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "normal"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    # infer the task type
    samples.attrs["task"] = "classification"

    # Get the data frame for the split.
    if split:
        split_value = split.value if isinstance(split, Split) else split
        samples = samples[samples.split == split_value].reset_index(drop=True)

    return samples
