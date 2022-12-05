"""Dataset that generates synthetic anomalies.

This dataset can be used when there is a lack of real anomalous data.
"""

import logging
import math
import os
import shutil
from pathlib import Path
from typing import Union

import albumentations as A
import cv2
import pandas as pd
from albumentations.pytorch import ToTensorV2
from pandas import DataFrame, Series

from anomalib.data.base.dataset import AnomalibDataset
from anomalib.data.utils import Augmenter, Split, read_image
from anomalib.pre_processing import PreProcessor

logger = logging.getLogger(__name__)


def make_synthetic_dataset(
    source_samples: DataFrame, im_dir: Union[Path, str], mask_dir: Union[Path, str], anomalous_ratio: float = 0.5
) -> DataFrame:
    """Convert a set of normal samples into a mixed set of normal and synthetic anomalous samples.

    The synthetic images will be saved to the file system in the specified root directory under <root>/images.
    For the synthetic anomalous images, the masks will be saved under <root>/ground_truth.

    Args:
        source_samples (DataFrame): Normal images that will be used as source for the synthetic anomalous images.
        im_dir (Union[Path, str]): Directory to which the synthetic anomalous image files will be written.
        mask_dir (Union[Path, str]): Directory to which the ground truth anomaly masks will be written.
        anomalous_ratio (float): Fraction of source samples that will be converted into anomalous samples.
    """
    assert 1 not in source_samples.label_index.values, "All source images must be normal."
    assert os.path.isdir(im_dir), f"{im_dir} is not a folder."
    assert os.path.isdir(mask_dir), f"{mask_dir} is not a folder"

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
        im_path = str(Path(im_dir) / file_name)
        cv2.imwrite(im_path, aug_im)
        # write mask
        mask = (mask.squeeze() * 255).numpy()
        mask_path = str(Path(mask_dir) / file_name)
        cv2.imwrite(mask_path, mask)
        out = dict(image_path=im_path, label="abnormal", label_index=1, mask_path=mask_path, split=Split.VAL)
        return Series(out)

    anomalous_samples = anomalous_samples.apply(augment, axis=1)

    samples = pd.concat([normal_samples, anomalous_samples], ignore_index=True)

    return samples


class SyntheticValidationSet(AnomalibDataset):
    """Dataset which reads synthetically generated anomalous images from a temporary folder.

    Args:
        task (str): Task type, either "classification" or "segmentation".
        pre_process (PreProcessor): Preprocessor object used to transform the input images.
        source_samples (DataFrame): Normal samples to which the anomalous augmentations will be applied.
    """

    def __init__(self, task: str, pre_process: PreProcessor, source_samples: DataFrame):
        super().__init__(task, pre_process)

        self.source_samples = source_samples

        # Files will be written to a temporary directory in the workdir, which is cleaned up after code execution
        self.root = Path("./.tmp/synthetic_anomaly")
        self.im_dir = self.root / "images"
        self.mask_dir = self.root / "ground_truth"

        # clean up any existing data that may be left over from previous run
        if os.path.exists(self.root):
            shutil.rmtree(self.root)

        # create directories
        os.makedirs(self.im_dir)
        os.makedirs(self.mask_dir)

        self.setup()

    @classmethod
    def from_dataset(cls, dataset):
        """Create a synthetic anomaly dataset from an existing dataset of normal images."""
        return cls(task=dataset.task, pre_process=dataset.pre_process, source_samples=dataset.samples)

    def _setup(self) -> None:
        """Create samples dataframe."""
        logger.info("Generating synthetic anomalous images for validation set")
        self.samples = make_synthetic_dataset(self.source_samples, self.im_dir, self.mask_dir, 0.5)

    def __del__(self):
        """Make sure the temporary directory is cleaned up when the dataset object is deleted."""
        shutil.rmtree(self.root)
