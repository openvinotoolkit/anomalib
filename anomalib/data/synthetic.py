"""Dataset that generates synthetic anomalies.

This dataset can be used when there is a lack of real anomalous data.
"""

import os
import tempfile
from pathlib import Path
from typing import Union

import albumentations as A
import cv2
import pandas as pd
from albumentations.pytorch import ToTensorV2
from pandas import DataFrame

from anomalib.data.base.dataset import AnomalibDataset
from anomalib.data.utils import read_image
from anomalib.models.draem.utils import Augmenter
from anomalib.pre_processing import PreProcessor


def make_synthetic_dataset(normal_samples: DataFrame, root: Union[Path, str]) -> DataFrame:
    """Convert a set of normal samples into a mixed set of normal and synthetic anomalous samples.

    The synthetic images will be saved to the file system in the specified root directory under <root>/images.
    For the synthetic anomalous images, the masks will be saved under <root>/ground_truth.

    Args:
        normal_samples (DataFrame): DataFrame describing a set of normal images.
        root (Union[Path, str]): Root directory to which the image files will be written.
    """
    im_dir = Path(root) / "images"
    mask_dir = Path(root) / "ground_truth"
    os.makedirs(im_dir)
    os.makedirs(mask_dir)

    # make fakes
    augmenter = Augmenter("./datasets/dtd")

    transform = A.Compose([A.ToFloat(), ToTensorV2()])

    new_samples_list = []
    for index, sample in normal_samples.iterrows():
        # load image
        im = read_image(sample.image_path)
        # to tensor
        im = transform(image=im)["image"].unsqueeze(0)
        # apply rand aug
        aug_im, mask = augmenter.augment_batch(im)
        #
        is_anomalous = mask.max() == 1
        # write image
        aug_im = (aug_im.squeeze().permute((1, 2, 0)) * 255).numpy()
        aug_im = cv2.cvtColor(aug_im, cv2.COLOR_RGB2BGR)
        im_path = im_dir / (str(index).zfill(3) + ".png")
        cv2.imwrite(str(im_path), aug_im)
        # write mask
        if is_anomalous:
            mask = (mask.squeeze() * 255).numpy()
            mask_path = mask_dir / (str(index).zfill(3) + ".png")
            cv2.imwrite(str(mask_path), mask)
        # update path in samples
        new_samples_list.append(
            dict(
                image_path=str(im_path),
                label="abnormal" if is_anomalous else "normal",
                label_index=1 if is_anomalous else 0,
                mask_path=str(mask_path) if is_anomalous else "",
                split=None,
            )
        )

    return pd.DataFrame(new_samples_list)


class SyntheticValidationSet(AnomalibDataset):
    """Dataset which reads synthetically generated anomalous images from a temporary folder.

    Args:
        task (str): Task type, either "classification" or "segmentation".
        pre_process (PreProcessor): Preprocessor object used to transform the input images.
        normal_samples (DataFrame): Normal samples to which the anomalous augmentations will be applied.
    """

    def __init__(self, task: str, pre_process: PreProcessor, normal_samples: DataFrame):
        super().__init__(task, pre_process)

        self.normal_samples = normal_samples
        self.tempfolder = tempfile.TemporaryDirectory(dir="./datasets")
        self.setup()

    @classmethod
    def from_dataset(cls, dataset):
        """Create a synthetic anomaly dataset from an existing dataset of normal images."""
        return cls(task=dataset.task, pre_process=dataset.pre_process, normal_samples=dataset.samples)

    def _setup(self) -> None:
        """Create samples dataframe."""
        self.samples = make_synthetic_dataset(self.normal_samples, self.tempfolder.name)

    def __del__(self):
        """Make sure the temporary directory is cleaned up when the dataset object is deleted."""
        self.tempfolder.cleanup()
