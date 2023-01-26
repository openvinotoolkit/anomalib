"""Dataset that generates synthetic anomalies.

This dataset can be used when there is a lack of real anomalous data.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
import shutil
from copy import deepcopy
from pathlib import Path
from tempfile import mkdtemp

import albumentations as A
import cv2
import pandas as pd
from albumentations.pytorch import ToTensorV2
from pandas import DataFrame, Series

from anomalib.data.base.dataset import AnomalibDataset
from anomalib.data.task_type import TaskType
from anomalib.data.utils import Augmenter, Split, read_image

logger = logging.getLogger(__name__)


ROOT = "./.tmp/synthetic_anomaly"


def make_synthetic_dataset(
    source_samples: DataFrame, image_dir: Path, mask_dir: Path, anomalous_ratio: float = 0.5
) -> DataFrame:
    """Convert a set of normal samples into a mixed set of normal and synthetic anomalous samples.

    The synthetic images will be saved to the file system in the specified root directory under <root>/images.
    For the synthetic anomalous images, the masks will be saved under <root>/ground_truth.

    Args:
        source_samples (DataFrame): Normal images that will be used as source for the synthetic anomalous images.
        image_dir (Path): Directory to which the synthetic anomalous image files will be written.
        mask_dir (Path): Directory to which the ground truth anomaly masks will be written.
        anomalous_ratio (float): Fraction of source samples that will be converted into anomalous samples.
    """
    assert 1 not in source_samples.label_index.values, "All source images must be normal."
    assert image_dir.is_dir(), f"{image_dir} is not a folder."
    assert mask_dir.is_dir(), f"{mask_dir} is not a folder"

    # filter relevant columns
    source_samples = source_samples.filter(["image_path", "label", "label_index", "mask_path", "split"])
    # randomly select samples for augmentation
    n_anomalous = int(anomalous_ratio * len(source_samples))
    anomalous_samples = source_samples.sample(n_anomalous)
    normal_samples = source_samples.drop(anomalous_samples.index)
    anomalous_samples = anomalous_samples.reset_index(drop=True)

    # initialize augmenter
    augmenter = Augmenter("./datasets/dtd", p_anomalous=1.0, beta=(0.01, 0.2))

    # initialize transform for source images
    transform = A.Compose([A.ToFloat(), ToTensorV2()])

    def augment(sample: Series) -> Series:
        """Helper function to apply synthetic anomalous augmentation to a sample from a dataframe.

        Reads an image, applies the augmentations, writes the augmented image and corresponding mask to the file system,
        and returns a new Series object with the updates labels and file locations.

        Args:
            sample (Series): DataFrame row containing info about the image that will be augmented.

        Returns:
            Series: DataFrame row with updated information about the augmented image.
        """
        # read and transform image
        image = read_image(sample.image_path)
        image = transform(image=image)["image"].unsqueeze(0)
        # apply anomalous perturbation
        aug_im, mask = augmenter.augment_batch(image)
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
        out = dict(image_path=str(im_path), label="abnormal", label_index=1, mask_path=str(mask_path), split=Split.VAL)
        return Series(out)

    anomalous_samples = anomalous_samples.apply(augment, axis=1)

    samples = pd.concat([normal_samples, anomalous_samples], ignore_index=True)

    return samples


class SyntheticAnomalyDataset(AnomalibDataset):
    """Dataset which reads synthetically generated anomalous images from a temporary folder.

    Args:
        task (str): Task type, either "classification" or "segmentation".
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
        source_samples (DataFrame): Normal samples to which the anomalous augmentations will be applied.
    """

    def __init__(self, task: TaskType, transform: A.Compose, source_samples: DataFrame) -> None:
        super().__init__(task, transform)

        self.source_samples = source_samples

        # Files will be written to a temporary directory in the workdir, which is cleaned up after code execution
        root = Path(ROOT)
        root.mkdir(parents=True, exist_ok=True)

        self.root = Path(mkdtemp(dir=root))
        self.im_dir = self.root / "abnormal"
        self.mask_dir = self.root / "ground_truth"

        # create directories
        self.im_dir.mkdir()
        self.mask_dir.mkdir()

        self._cleanup = True  # flag that determines if temp dir is cleaned up when instance is deleted
        self.setup()

    @classmethod
    def from_dataset(cls, dataset: AnomalibDataset) -> SyntheticAnomalyDataset:
        """Create a synthetic anomaly dataset from an existing dataset of normal images.

        Args:
            dataset (AnomalibDataset): Dataset consisting of only normal images that will be converrted to a synthetic
                anomalous dataset with a 50/50 normal anomalous split.
        """
        return cls(task=dataset.task, transform=dataset.transform, source_samples=dataset.samples)

    def __copy__(self) -> SyntheticAnomalyDataset:
        """Returns a shallow copy of the dataset object and prevents cleanup when original object is deleted."""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        self._cleanup = False
        return new

    def __deepcopy__(self, _memo: dict) -> SyntheticAnomalyDataset:
        """Returns a deep copy of the dataset object and prevents cleanup when original object is deleted."""
        cls = self.__class__
        new = cls.__new__(cls)
        for key, value in self.__dict__.items():
            setattr(new, key, deepcopy(value))
        self._cleanup = False
        return new

    def _setup(self) -> None:
        """Create samples dataframe."""
        logger.info("Generating synthetic anomalous images for validation set")
        self.samples = make_synthetic_dataset(self.source_samples, self.im_dir, self.mask_dir, 0.5)

    def __del__(self) -> None:
        """Make sure the temporary directory is cleaned up when the dataset object is deleted."""
        if self._cleanup:
            shutil.rmtree(self.root)
