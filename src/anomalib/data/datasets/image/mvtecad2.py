"""MVTec AD 2 Dataset.

This module provides PyTorch Dataset implementation for the MVTec AD 2 dataset.
The dataset contains 8 categories of industrial objects with both normal and
anomalous samples. Each category includes RGB images and pixel-level ground truth
masks for anomaly segmentation.

The dataset provides three different test sets:
    - Public test set (test_public/): Contains both normal and anomalous samples with ground truth masks
    - Private test set (test_private/): Contains unseen test samples without ground truth
    - Private mixed test set (test_private_mixed/): Contains unseen test samples
        with mixed anomalies without ground truth

The public test set is used for standard evaluation, while the private test sets
are used for real-world evaluation scenarios where ground truth is not available.

License:
    MVTec AD 2 dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    Lars Heckler-Kram, Jan-Hendrik Neudeck, Ulla Scheler, Rebecca König, Carsten Steger:
    The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection.
    arXiv preprint, 2024 (to appear).
"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from enum import Enum
from pathlib import Path

from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import Split, validate_path


class TestType(str, Enum):
    """Type of test set to use.

    The MVTec AD 2 dataset provides three different test sets:
        - PUBLIC: Test set with ground truth masks for facilitating local testing and initial performance estimation
        - PRIVATE: Official unseen test set without ground truth for entering the leaderboard
        - PRIVATE_MIXED: Official unseen test set captured under seen and unseen lighting conditions (mixed randomly)

    Official evaluation server: https://benchmark.mvtec.com/
    """

    PUBLIC = "public"  # Test set with ground truth for local evaluation
    PRIVATE = "private"  # Official private test set without ground truth
    PRIVATE_MIXED = "private_mixed"  # Official private test set with mixed lighting conditions


IMG_EXTENSIONS = (".png", ".PNG")
CATEGORIES = (
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
)


class MVTecAD2Dataset(AnomalibDataset):
    """MVTec AD 2 dataset class.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/MVTec_AD_2"``.
        category (str): Category name, e.g. ``"sheet_metal"``.
            Defaults to ``"sheet_metal"``.
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None): Dataset split - usually ``Split.TRAIN``, ``Split.VAL``,
            or ``Split.TEST``. Defaults to ``None``.
        test_type (str | TestType): Type of test set to use - only used when split is ``Split.TEST``:
            - ``"public"``: Test set with ground truth for local evaluation and initial performance estimation
            - ``"private"``: Official test set without ground truth for leaderboard submission
            - ``"private_mixed"``: Official test set with mixed lighting conditions (seen and unseen lighting)
            Defaults to ``TestType.PUBLIC``.

    Example:
        Create training dataset::

            >>> from pathlib import Path
            >>> dataset = MVTecAD2Dataset(
            ...     root=Path("./datasets/MVTec_AD_2"),
            ...     category="sheet_metal",
            ...     split="train"
            ... )

        Create validation dataset::

            >>> val_dataset = MVTecAD2Dataset(
            ...     root=Path("./datasets/MVTec_AD_2"),
            ...     category="sheet_metal",
            ...     split="val"
            ... )

        Create test datasets::

            >>> # Public test set (with ground truth)
            >>> test_dataset = MVTecAD2Dataset(
            ...     root=Path("./datasets/MVTec_AD_2"),
            ...     category="sheet_metal",
            ...     split="test",
            ...     test_type="public"
            ... )

            >>> # Private test set (without ground truth)
            >>> private_dataset = MVTecAD2Dataset(
            ...     root=Path("./datasets/MVTec_AD_2"),
            ...     category="sheet_metal",
            ...     split="test",
            ...     test_type="private"
            ... )

            >>> # Private mixed test set (without ground truth)
            >>> mixed_dataset = MVTecAD2Dataset(
            ...     root=Path("./datasets/MVTec_AD_2"),
            ...     category="sheet_metal",
            ...     split="test",
            ...     test_type="private_mixed"
            ... )

    Notes:
        - The public test set contains both normal and anomalous samples with ground truth masks
        - Private test sets (private and private_mixed) contain samples without ground truth
        - Private test samples are labeled as "unknown" with label_index=-1
    """

    def __init__(
        self,
        root: Path | str = "./datasets/MVTec_AD_2",
        category: str = "sheet_metal",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
        test_type: TestType | str = TestType.PUBLIC,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.root_category = Path(root) / Path(category)
        self.split = split
        self.test_type = TestType(test_type) if isinstance(test_type, str) else test_type
        self.samples = make_mvtec2_dataset(
            self.root_category,
            split=self.split,
            test_type=self.test_type,
            extensions=IMG_EXTENSIONS,
        )


def make_mvtec2_dataset(
    root: str | Path,
    split: str | Split | None = None,
    test_type: TestType = TestType.PUBLIC,
    extensions: Sequence[str] | None = None,
) -> DataFrame:
    """Create MVTec AD 2 samples by parsing the data directory structure.

    The files are expected to follow this structure::

        root/
        ├── test_private/
        ├── test_private_mixed/
        ├── test_public/
        │   ├── bad/
        │   ├── good/
        │   └── ground_truth/
        │       └── bad/
        ├── train/
        │   └── good/
        └── validation/
            └── good/

    Args:
        root (str | Path): Path to the dataset root directory
        split (str | Split | None, optional): Dataset split (train, val, test). Defaults to None.
        test_type (TestType, optional): Type of test set to use for testing:
            - PUBLIC: Test set with ground truth (for local evaluation)
            - PRIVATE: Official test set without ground truth (for leaderboard)
            - PRIVATE_MIXED: Official test set with mixed lighting conditions (for leaderboard)
            Defaults to TestType.PUBLIC.
        extensions (Sequence[str] | None, optional): Image extensions to include. Defaults to None.

    Returns:
        DataFrame: Dataset samples with columns:
            - path: Base path to dataset
            - split: Dataset split (train/test)
            - label: Class label
            - image_path: Path to image file
            - mask_path: Path to mask file (if available)
            - label_index: Numeric label (0=normal, 1=abnormal)

    Example:
        >>> root = Path("./datasets/MVTec_AD_2/sheet_metal")
        >>> samples = make_mvtec2_dataset(root, split="train")
        >>> samples.head()
           path                split label image_path           mask_path label_index
        0  datasets/MVTec_AD_2 train good  [...]/good/105.png           0
        1  datasets/MVTec_AD_2 train good  [...]/good/017.png           0

    Raises:
        RuntimeError: If no valid images are found
        MisMatchError: If anomalous images and masks don't match
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = validate_path(root)
    samples_list: list[tuple[str, str, str, str, str | None, int]] = []

    # Get all image files
    image_files = [f for f in root.glob("**/*") if f.suffix in extensions]
    if not image_files:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    # Process training samples (only normal)
    train_path = root / "train" / "good"
    if train_path.exists():
        train_samples = [
            (str(root), "train", "good", str(f), None, 0) for f in train_path.glob(f"*[{''.join(extensions)}]")
        ]
        samples_list.extend(train_samples)

    # Process validation samples (only normal)
    val_path = root / "validation" / "good"
    if val_path.exists():
        val_samples = [(str(root), "val", "good", str(f), None, 0) for f in val_path.glob(f"*[{''.join(extensions)}]")]
        samples_list.extend(val_samples)

    # Process test samples based on test_type
    if test_type == TestType.PUBLIC:
        test_path = root / "test_public"
        if test_path.exists():
            # Normal test samples
            test_normal_path = test_path / "good"
            test_normal_samples = [
                (str(root), "test", "good", str(f), None, 0) for f in test_normal_path.glob(f"*[{''.join(extensions)}]")
            ]
            samples_list.extend(test_normal_samples)

            # Abnormal test samples
            test_abnormal_path = test_path / "bad"
            if test_abnormal_path.exists():
                for image_path in test_abnormal_path.glob(f"*[{''.join(extensions)}]"):
                    # Add _mask suffix to the filename
                    mask_name = image_path.stem + "_mask" + image_path.suffix
                    mask_path = root / "test_public" / "ground_truth" / "bad" / mask_name
                    if not mask_path.exists():
                        msg = f"Missing mask for anomalous image: {image_path}"
                        raise MisMatchError(msg)
                    samples_list.append(
                        (str(root), "test", "bad", str(image_path), str(mask_path), 1),
                    )
    elif test_type == TestType.PRIVATE:
        test_path = root / "test_private"
        if test_path.exists():
            # All samples in private test set are treated as unknown
            test_samples = [
                (str(root), "test", "unknown", str(f), None, -1) for f in test_path.glob(f"*[{''.join(extensions)}]")
            ]
            samples_list.extend(test_samples)
    elif test_type == TestType.PRIVATE_MIXED:
        test_path = root / "test_private_mixed"
        if test_path.exists():
            # All samples in private mixed test set are treated as unknown
            test_samples = [
                (str(root), "test", "unknown", str(f), None, -1) for f in test_path.glob(f"*[{''.join(extensions)}]")
            ]
            samples_list.extend(test_samples)

    samples = DataFrame(
        samples_list,
        columns=["path", "split", "label", "image_path", "mask_path", "label_index"],
    )

    # Filter by split if specified
    if split:
        split = Split(split) if isinstance(split, str) else split
        samples = samples[samples.split == split.value]

    samples.attrs["task"] = "segmentation"
    return samples
