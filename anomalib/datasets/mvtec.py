"""
MVTec
This script contains PyTorch Dataset, Dataloader and PyTorch Lightning
    DataModule for the MVTec dataset. If the dataset is not on the file
    system, the script downloads and extracts the dataset and create
    PyTorch data objects.
"""

import logging
import random
import tarfile
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union
from urllib.request import urlretrieve
from warnings import warn

import pandas as pd
import torch
import torchvision.transforms as T
from pandas.core.frame import DataFrame
from PIL import Image
from pytorch_lightning.core.datamodule import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import VisionDataset
from torchvision.transforms import Compose

from anomalib.utils.download_progress_bar import DownloadProgressBar

logger = logging.getLogger(name="Dataset: MVTec")
logger.setLevel(logging.DEBUG)

__all__ = ["MVTec", "MVTecDataModule"]


def split_normal_images_in_train_set(samples: DataFrame, split_ratio: float = 0.1, seed: int = 0) -> DataFrame:
    """
    split_normal_images_in_train_set
        This function splits the normal images in training set and assigns the
        values to the test set. This is particularly useful especially when the
        test set does not contain any normal images.

        This is important because when the test set doesn't have any normal images,
        AUC computation fails due to having single class.

    Args:
        samples (DataFrame): Dataframe containing dataset info such as filenames, splits etc.
        split_ratio (float, optional): Train-Test normal image split ratio. Defaults to 0.1.
        seed (int, optional): Random seed to ensure reproducibility. Defaults to 0.

    Returns:
        DataFrame: Output dataframe where the part of the training set is assigned to test set.
    """

    random.seed(seed)

    normal_train_image_indices = samples.index[(samples.split == "train") & (samples.label == "good")].to_list()
    num_normal_train_images = len(normal_train_image_indices)
    num_normal_valid_images = int(num_normal_train_images * split_ratio)

    indices_to_split_from_train_set = random.sample(population=normal_train_image_indices, k=num_normal_valid_images)
    samples.loc[indices_to_split_from_train_set, "split"] = "test"

    return samples


def make_mvtec_dataset(path: Path, split: str = "train", split_ratio: float = 0.1, seed: int = 0) -> DataFrame:
    """
    This function creates MVTec samples by parsing the MVTec data file structure, based on the following
    structure:
        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/mask_filename.png

    This function creates a dataframe to store the parsed information based on the following format:
    |---|---------------|-------|---------|---------------|---------------------------------------|-------------|
    |   | path          | split | label   | image_path    | mask_path                             | label_index |
    |---|---------------|-------|---------|---------------|---------------------------------------|-------------|
    | 0 | datasets/name |  test |  defect |  filename.png | ground_truth/defect/filename_mask.png | 1           |
    |---|---------------|-------|---------|---------------|---------------------------------------|-------------|

    Args:
        path (Path): Path to dataset
        split (str, optional): Dataset split (ie., either train or test). Defaults to "train".
        split_ratio (float, optional): Ratio to split normal training images and add to the
                                       test set in case test set doesn't contain any normal images.
                                       Defaults to 0.1.
        seed (int, optional): Random seed to ensure reproducibility when splitting. Defaults to 0.

    Example:
        The following example shows how to get training samples from MVTec bottle category:

        >>> root = Path('./MVTec')
        >>> category = 'bottle'
        >>> path = root / category
        >>> path
        PosixPath('MVTec/bottle')

        >>> samples = make_mvtec_dataset(path, split='train', split_ratio=0.1, seed=0)
        >>> samples.head()
           path         split label image_path                           mask_path                   label_index
        0  MVTec/bottle train good MVTec/bottle/train/good/105.png MVTec/bottle/ground_truth/good/105_mask.png 0
        1  MVTec/bottle train good MVTec/bottle/train/good/017.png MVTec/bottle/ground_truth/good/017_mask.png 0
        2  MVTec/bottle train good MVTec/bottle/train/good/137.png MVTec/bottle/ground_truth/good/137_mask.png 0
        3  MVTec/bottle train good MVTec/bottle/train/good/152.png MVTec/bottle/ground_truth/good/152_mask.png 0
        4  MVTec/bottle train good MVTec/bottle/train/good/109.png MVTec/bottle/ground_truth/good/109_mask.png 0

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)
    """
    samples_list = [(str(path),) + filename.parts[-3:] for filename in path.glob("**/*.png")]
    if len(samples_list) == 0:
        raise RuntimeError(f"Found 0 images in {path}")

    samples = pd.DataFrame(samples_list, columns=["path", "split", "label", "image_path"])
    samples = samples[samples.split != "ground_truth"]

    # Create mask_path column
    samples["mask_path"] = (
        samples.path
        + "/ground_truth/"
        + samples.label
        + "/"
        + samples.image_path.str.rstrip("png").str.rstrip(".")
        + "_mask.png"
    )

    # Modify image_path column by converting to absolute path
    samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    # Split the normal images in training set if test set doesn't
    # contain any normal images. This is needed because AUC score
    # cannot be computed based on 1-class
    if sum((samples.split == "test") & (samples.label == "good")) == 0:
        samples = split_normal_images_in_train_set(samples, split_ratio, seed)

    # Good images don't have mask
    samples.loc[(samples.split == "test") & (samples.label == "good"), "mask_path"] = ""

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "good"), "label_index"] = 0
    samples.loc[(samples.label != "good"), "label_index"] = 1
    samples.label_index = samples.label_index.astype(int)

    # Get the data frame for the split.
    samples = samples[samples.split == split]
    samples = samples.reset_index(drop=True)

    return samples


def get_image_transforms(image_size: Union[Sequence, int], crop_size: Union[Sequence, int]) -> T.Compose:
    """
    Get default ImageNet image transformations.

    Returns:
        T.Compose: List of imagenet transformations.

    """
    crop_size = image_size if crop_size is None else crop_size
    transform = T.Compose(
        [
            T.Resize(image_size, Image.ANTIALIAS),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform


def get_mask_transforms(image_size: Union[Sequence, int], crop_size: Union[Sequence, int]) -> T.Compose:
    """
    Get default ImageNet transformations for the ground-truth image masks.

    Returns:
      T.Compose: List of imagenet transformations.

    """
    crop_size = image_size if crop_size is None else crop_size
    transform = Compose(
        [
            T.Resize(image_size, Image.NEAREST),
            T.CenterCrop(crop_size),
            T.ToTensor(),
        ]
    )
    return transform


class MVTec(VisionDataset):
    """
    MVTec PyTorch Dataset
    """

    def __init__(
        self,
        root: Union[Path, str],
        category: str,
        image_transforms: Callable,
        mask_transforms: Callable,
        train: bool = True,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=image_transforms, target_transform=mask_transforms)
        self.root = Path(root) if isinstance(root, str) else root
        self.category: str = category
        self.split = "train" if train else "test"
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        self.download = download

        if self.download:
            self._download()

        self.samples = make_mvtec_dataset(path=self.root / category, split=self.split)

    def _download(self) -> None:
        """
        Download the MVTec dataset
        """
        if (self.root / self.category).is_dir():
            logger.warning("Dataset directory exists.")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            dataset_name = "mvtec_anomaly_detection.tar.xz"
            self.filename = self.root / dataset_name

            logger.info("Downloading MVTec Dataset")
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dataset_name) as progress_bar:
                urlretrieve(
                    url=f"ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/{dataset_name}",
                    filename=self.filename,
                    reporthook=progress_bar.update_to,
                )

            self._extract()
            self._clean()

    def _extract(self) -> None:
        """
        Extract MVTec Dataset
        """
        logger.info("Extracting MVTec dataset")
        with tarfile.open(self.filename) as file:
            file.extractall(self.root)

    def _clean(self) -> None:
        """
        Cleanup MVTec Dataset tar file.
        """
        logger.info("Cleaning up the tar file")
        self.filename.unlink()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Union[Dict[str, Tensor], Dict[str, Union[str, Tensor]]]:
        image_path = self.samples.image_path[index]
        mask_path = self.samples.mask_path[index]
        label_index = self.samples.label_index[index]

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transforms(image)

        if self.split == "train":
            sample = {"image": image_tensor}
        else:
            if label_index == 0:
                mask = Image.new(mode="1", size=image.size)
            else:
                mask = Image.open(mask_path).convert(mode="1")

            mask_tensor = self.mask_transforms(mask).to(torch.uint8)

            sample = {
                "image_path": image_path,
                "mask_path": mask_path,
                "image": image_tensor,
                "label": label_index,
                "mask": mask_tensor,
            }

        return sample


class MVTecDataModule(LightningDataModule):
    """
    MVTec Lightning Data Module
    """

    def __init__(
        self,
        root: str,
        category: str,
        image_size: Union[Sequence, int],
        crop_size: Union[Sequence, int],
        train_batch_size: int,
        test_batch_size: int,
        num_workers: int,
        image_transforms: Optional[Callable] = None,
        mask_transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        warn("MVTec DataModule will be deprecated. Use AnomalyDataModule instead")

        self.root = root if isinstance(root, Path) else Path(root)
        self.category = category
        self.dataset_path = self.root / self.category
        self.image_size = image_size
        self.crop_size = crop_size
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.image_transforms = (
            image_transforms if image_transforms is not None else get_image_transforms(image_size, crop_size)
        )
        self.mask_transforms = (
            mask_transforms if mask_transforms is not None else get_mask_transforms(image_size, crop_size)
        )

        self.train_data: Dataset
        self.val_data: Dataset

    def prepare_data(self):
        """
        Prepare MVTec Dataset
        """

        # Train
        MVTec(
            root=self.root,
            category=self.category,
            train=True,
            image_transforms=self.image_transforms,
            mask_transforms=self.mask_transforms,
            download=True,
        )

        # Test
        MVTec(
            root=self.root,
            category=self.category,
            train=False,
            image_transforms=self.image_transforms,
            mask_transforms=self.mask_transforms,
            download=True,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup train, validation and test data.

        Args:
          stage: Optional[str]:  (Default value = None)

        """
        self.val_data = MVTec(
            root=self.root,
            category=self.category,
            train=False,
            image_transforms=self.image_transforms,
            mask_transforms=self.mask_transforms,
        )
        if stage in (None, "fit"):
            self.train_data = MVTec(
                root=self.root,
                category=self.category,
                train=True,
                image_transforms=self.image_transforms,
                mask_transforms=self.mask_transforms,
            )

    def train_dataloader(self) -> DataLoader:
        """Get train dataloader"""
        return DataLoader(
            self.train_data, shuffle=False, batch_size=self.train_batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader"""
        return DataLoader(self.val_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader"""
        return DataLoader(self.val_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers)
