"""Dataset that generates synthetic anomalies.

This module provides functionality to generate synthetic anomalies when real
anomalous data is scarce or unavailable. It includes:

- A dataset class that generates synthetic anomalies from normal images
- Functions to convert normal samples into synthetic anomalous samples
- Perlin noise-based anomaly generation
- Temporary file management for synthetic data

Example:
    >>> from anomalib.data.utils.synthetic import SyntheticAnomalyDataset
    >>> # Create synthetic dataset from normal samples
    >>> synthetic_dataset = SyntheticAnomalyDataset(
    ...     transform=transforms,
    ...     source_samples=normal_samples
    ... )
    >>> len(synthetic_dataset)  # 50/50 normal/anomalous split
    200
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import shutil
from copy import deepcopy
from pathlib import Path
from tempfile import mkdtemp

import cv2
import pandas as pd
from pandas import DataFrame, Series
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.utils import Split, read_image
from anomalib.data.utils.generators.perlin import PerlinAnomalyGenerator

logger = logging.getLogger(__name__)


ROOT = "./.tmp/synthetic_anomaly"


def make_synthetic_dataset(
    source_samples: DataFrame,
    image_dir: Path,
    mask_dir: Path,
    anomalous_ratio: float = 0.5,
) -> DataFrame:
    """Convert normal samples into a mixed set with synthetic anomalies.

    The function generates synthetic anomalous images and their corresponding
    masks by applying Perlin noise-based perturbations to normal images.

    Args:
        source_samples: DataFrame containing normal images used as source for
            synthetic anomalies. Must contain columns: ``image_path``,
            ``label``, ``label_index``, ``mask_path``, and ``split``.
        image_dir: Directory where synthetic anomalous images will be saved.
        mask_dir: Directory where ground truth anomaly masks will be saved.
        anomalous_ratio: Fraction of source samples to convert to anomalous
            samples. Defaults to ``0.5``.

    Returns:
        DataFrame containing both normal and synthetic anomalous samples.

    Raises:
        ValueError: If source samples contain any anomalous images.
        NotADirectoryError: If ``image_dir`` or ``mask_dir`` is not a directory.

    Example:
        >>> df = make_synthetic_dataset(
        ...     source_samples=normal_df,
        ...     image_dir=Path("./synthetic/images"),
        ...     mask_dir=Path("./synthetic/masks"),
        ...     anomalous_ratio=0.3
        ... )
        >>> len(df[df.label == "abnormal"])  # 30% are anomalous
        30
    """
    if 1 in source_samples.label_index.to_numpy():
        msg = "All source images must be normal."
        raise ValueError(msg)

    if not image_dir.is_dir():
        msg = f"{image_dir} is not a folder."
        raise NotADirectoryError(msg)

    if not mask_dir.is_dir():
        msg = f"{mask_dir} is not a folder."
        raise NotADirectoryError(msg)

    # filter relevant columns
    source_samples = source_samples.filter(["image_path", "label", "label_index", "mask_path", "split"])
    # randomly select samples for augmentation
    n_anomalous = int(anomalous_ratio * len(source_samples))
    anomalous_samples = source_samples.sample(n_anomalous)
    normal_samples = source_samples.drop(anomalous_samples.index)
    anomalous_samples = anomalous_samples.reset_index(drop=True)

    # initialize augmenter
    augmenter = PerlinAnomalyGenerator(
        anomaly_source_path="./datasets/dtd",
        probability=1.0,
        blend_factor=(0.01, 0.2),
    )

    def augment(sample: Series) -> Series:
        """Apply synthetic anomalous augmentation to a sample.

        Args:
            sample: DataFrame row containing image information.

        Returns:
            Series containing updated information about the augmented image.
        """
        # read and transform image
        image = read_image(sample.image_path, as_tensor=True)
        # apply anomalous perturbation
        aug_im, mask = augmenter(image)
        # target file name with leading zeros
        file_name = f"{str(sample.name).zfill(int(math.log10(n_anomalous)) + 1)}.png"
        # write image
        aug_im = (aug_im.squeeze().permute((1, 2, 0)) * 255).numpy()
        aug_im = cv2.cvtColor(aug_im, cv2.COLOR_RGB2BGR)
        im_path = image_dir / file_name
        cv2.imwrite(str(im_path), aug_im)
        # write mask
        mask = (mask.squeeze() * 255).numpy()
        mask_path = mask_dir / file_name
        cv2.imwrite(str(mask_path), mask)
        out = {
            "image_path": str(im_path),
            "label": "abnormal",
            "label_index": 1,
            "mask_path": str(mask_path),
            "split": Split.VAL,
        }
        return Series(out)

    anomalous_samples = anomalous_samples.apply(augment, axis=1)

    return pd.concat([normal_samples, anomalous_samples], ignore_index=True)


class SyntheticAnomalyDataset(AnomalibDataset):
    """Dataset for generating and managing synthetic anomalies.

    The dataset creates synthetic anomalous images by applying Perlin
    noise-based perturbations to normal images. The synthetic images are
    stored in a temporary directory that is cleaned up when the dataset
    object is deleted.

    Args:
        augmentations (Transform | None): Transform object describing the input data augmentations.
        source_samples: DataFrame containing normal samples used as source for
            synthetic anomalies.
        dataset_name: str dataset name for path of temporary anomalous samples

    Example:
        >>> transform = Compose([...])
        >>> dataset = SyntheticAnomalyDataset(
        ...     transform=transform,
        ...     source_samples=normal_df,
        ...     dataset_name="synthetic"
        ... )
        >>> len(dataset)  # 50/50 normal/anomalous split
        100
    """

    def __init__(self, augmentations: Transform | None, source_samples: DataFrame, dataset_name: str) -> None:
        super().__init__(augmentations=augmentations)

        self.source_samples = source_samples

        # Files will be written to a temporary directory in the workdir
        root = Path(ROOT) / dataset_name
        root.mkdir(parents=True, exist_ok=True)

        self.root = Path(mkdtemp(dir=root))
        self.im_dir = self.root / "abnormal"
        self.mask_dir = self.root / "ground_truth"

        # create directories
        self.im_dir.mkdir()
        self.mask_dir.mkdir()

        self._cleanup = True  # flag that determines if temp dir is cleaned up
        self.samples = make_synthetic_dataset(
            self.source_samples,
            self.im_dir,
            self.mask_dir,
            0.5,
        )

        self.samples.attrs["task"] = "segmentation"

    @classmethod
    def from_dataset(
        cls: type["SyntheticAnomalyDataset"],
        dataset: AnomalibDataset,
    ) -> "SyntheticAnomalyDataset":
        """Create synthetic dataset from existing dataset of normal images.

        Args:
            dataset: Dataset containing only normal images to convert into a
                synthetic dataset with 50/50 normal/anomalous split.

        Returns:
            New synthetic anomaly dataset.

        Example:
            >>> normal_dataset = Dataset(...)
            >>> synthetic = SyntheticAnomalyDataset.from_dataset(normal_dataset)
        """
        return cls(augmentations=dataset.augmentations, source_samples=dataset.samples, dataset_name=dataset.name)

    def __copy__(self) -> "SyntheticAnomalyDataset":
        """Return shallow copy and prevent cleanup of original.

        Returns:
            Shallow copy of the dataset object.
        """
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        self._cleanup = False
        return new

    def __deepcopy__(self, _memo: dict) -> "SyntheticAnomalyDataset":
        """Return deep copy and prevent cleanup of original.

        Args:
            _memo: Memo dictionary used by deepcopy.

        Returns:
            Deep copy of the dataset object.
        """
        cls = self.__class__
        new = cls.__new__(cls)
        for key, value in self.__dict__.items():
            setattr(new, key, deepcopy(value))
        self._cleanup = False
        return new

    def __del__(self) -> None:
        """Clean up temporary directory when dataset object is deleted."""
        if self._cleanup:
            shutil.rmtree(self.root)
