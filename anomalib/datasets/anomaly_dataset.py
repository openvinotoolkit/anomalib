"""
Anomaly Dataset
This script contains PyTorch Dataset, Dataloader and PyTorch Lightning
    DataModule for the Anomaly dataset. If the dataset is not on the file
    system, the script downloads and extracts the dataset from URL and create
    PyTorch data objects.
"""
import abc
import logging
import random
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.request import urlretrieve

import albumentations as a
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, ListConfig
from pandas.core.frame import DataFrame
from pytorch_lightning.core.datamodule import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets.folder import VisionDataset

from anomalib.utils.download_progress_bar import DownloadProgressBar
from .parser import PascalVocReader

logger = logging.getLogger(name="Dataset: Anomaly")

__all__ = [
    "BaseAnomalyDataset",
    "AnomalyTrainDS",
    "AnomalyTestSegmentationDS",
    "AnomalyTestClassificationDS",
    "AnomalyTestDetectionDS",
    "AnomalyDataModule",
]


def read_image(path: str) -> np.ndarray:
    """
    read_image
        reads image from disk in RGB format
    Args:
        path: path to the image file

    Returns:
        image as numpy array
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


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

    if seed > 0:
        random.seed(seed)

    normal_train_image_indices = samples.index[(samples.split == "train") & (samples.label == "good")].to_list()
    num_normal_train_images = len(normal_train_image_indices)
    num_normal_valid_images = int(num_normal_train_images * split_ratio)

    indices_to_split_from_train_set = random.sample(population=normal_train_image_indices, k=num_normal_valid_images)
    samples.loc[indices_to_split_from_train_set, "split"] = "test"

    return samples


def make_dataset(path: Path, split_ratio: float = 0.1, seed: int = 0) -> DataFrame:
    """
    This function creates MVTec samples by parsing the MVTec data file structure, based on the following
    structure:
        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/mask_filename.png

    This function creates a dataframe to store the parsed information based on the following format:
    |---|---------------|---------|---------------|---------------------------------------|-------------|
    |   | path          | label   | image_path    | mask_path                             | label_index |
    |---|---------------|---------|---------------|---------------------------------------|-------------|
    | 0 | datasets/name |  defect |  filename.png | ground_truth/defect/filename_mask.png | 1           |
    |---|---------------|---------|---------------|---------------------------------------|-------------|

    Args:
        path (Path): Path to dataset
        split_ratio (float, optional): Ratio to split normal training images and add to the
                                       test set in case test set doesn't contain any normal images.
                                       Defaults to 0.1.
        seed (int, optional): Random seed to ensure reproducibility when splitting. Defaults to 0.

    Example:
        The following example shows how to get training samples from MVTec bottle category:

        >>> root = Path('./MVTec')
        >>> category = 'bottle'
        >>> data_path = root / category
        >>> data_path
        PosixPath('MVTec/bottle')

        >>> samples_df = make_dataset(data_path, split_ratio=0.1, seed=0)
        >>> samples_df.head()
           path         split label image_path                           mask_path                   label_index
        0  MVTec/bottle train good MVTec/bottle/train/good/105.png MVTec/bottle/ground_truth/good/105_mask.png 0
        1  MVTec/bottle train good MVTec/bottle/train/good/017.png MVTec/bottle/ground_truth/good/017_mask.png 0
        2  MVTec/bottle train good MVTec/bottle/train/good/137.png MVTec/bottle/ground_truth/good/137_mask.png 0
        3  MVTec/bottle train good MVTec/bottle/train/good/152.png MVTec/bottle/ground_truth/good/152_mask.png 0
        4  MVTec/bottle train good MVTec/bottle/train/good/109.png MVTec/bottle/ground_truth/good/109_mask.png 0

    Returns:
        DataFrame: an output dataframe containing all samples
    """
    samples_list = [(str(path),) + filename.parts[3:] for filename in path.glob("**/*.png")]
    if len(samples_list) == 0:
        raise RuntimeError(f"Found 0 images in {path}")

    samples = pd.DataFrame(samples_list, columns=["path", "split", "label", "image_path"])
    samples = samples[samples.split != "ground_truth"]

    # Create mask_path column
    samples["target_path"] = (
        samples.path + "/ground_truth/" + samples.label + "/" + samples.image_path.str.rstrip("png").str.rstrip(".")
    )

    # Modify image_path column by converting to absolute path
    samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    # Split the normal images in training set if test set doesn't
    # contain any normal images. This is needed because AUC score
    # cannot be computed based on 1-class
    if sum((samples.split == "test") & (samples.label == "good")) == 0:
        samples = split_normal_images_in_train_set(samples, split_ratio, seed)

    # Good images don't have mask
    samples.loc[(samples.split == "test") & (samples.label == "good"), "target_path"] = ""

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "good"), "label_index"] = 0
    samples.loc[(samples.label != "good"), "label_index"] = 1
    samples.label_index = samples.label_index.astype(int)

    return samples


def get_transform(aug_prob: float, t: Union[DictConfig, ListConfig]) -> a.Compose:
    """
    get_transforms:
        Build a pipeline of image transforms each with different probability defined in config file

    return:
        a.compose: composed augmentation pipeline
    """

    return a.Compose(
        [
            a.Resize(t.image_size[0], t.image_size[1], always_apply=True),
            a.OneOf(
                [
                    a.RandomRotate90(),
                    a.HorizontalFlip(),
                    # Transpose(),
                ],
                p=t.rotate_flip_p,
            ),
            a.GaussNoise(p=t.gauss_noise_p),
            a.OneOf(
                [
                    a.MotionBlur(p=t.blur.motion_blur_p),
                    a.MedianBlur(blur_limit=t.blur.median_blur.blur_limit, p=t.blur.median_blur.p),
                    a.Blur(blur_limit=t.blur.blur.blur_limit, p=t.blur.blur.p),
                ],
                p=t.blur.p,
            ),
            a.ShiftScaleRotate(
                shift_limit=t.shift_scale_rotate.shift_limit,
                scale_limit=t.shift_scale_rotate.scale_limit,
                rotate_limit=t.shift_scale_rotate.rotate_limit,
                p=t.shift_scale_rotate.p,
            ),
            a.OneOf(
                [
                    a.OpticalDistortion(p=t.geometric_transforms.optical_distortion_p),
                    a.GridDistortion(p=t.geometric_transforms.grid_distortion_p),
                    a.PiecewiseAffine(p=t.geometric_transforms.affine_p),
                ],
                p=t.geometric_transforms.p,
            ),
            a.OneOf(
                [
                    a.CLAHE(clip_limit=t.image_adjustments.clahe_clip_limit),
                    a.Sharpen(),
                    a.Emboss(),
                    a.RandomBrightnessContrast(),
                    a.HueSaturationValue(),
                ],
                p=t.image_adjustments.p,
            ),
            a.ImageCompression(
                p=t.image_compression.p,
                quality_lower=t.image_compression.quality_lower,
                quality_upper=t.image_compression.quality_upper,
            ),
            a.CenterCrop(t.crop_size[0], t.crop_size[1], always_apply=True),
            a.Normalize(mean=t.normalize.mean, std=t.normalize.std, always_apply=True),
            a.ToGray(always_apply=True),
            ToTensorV2(always_apply=True),
        ],
        p=aug_prob,
    )


class BaseAnomalyDataset(VisionDataset):
    """
    Anomaly PyTorch Dataset
    """

    _TARGET_FILE_EXT: str
    _SPLIT: str

    def __init__(
        self,
        root: Union[Path, str],
        category: Path,
        label_format: Optional[str] = None,
        transform_params: Optional[Union[DictConfig, ListConfig]] = None,
        download: bool = False,
        download_url: Optional[str] = None,
        samples: Optional[DataFrame] = None,
    ) -> None:
        super().__init__(root, transform=None)
        self.root = Path(root) if isinstance(root, str) else root
        self.category: Path = category
        self.label_format = label_format
        self.download_url = download_url

        if transform_params is None:
            raise ValueError(f"Transform parameters not defined!")

        aug_prob: float = 0.0
        if self._SPLIT == "train":
            aug_prob = transform_params.p

        self.transform = get_transform(aug_prob, transform_params)

        if download:
            self._download()

        if samples is not None:
            self.samples = samples

        if download is False and len(self.samples) == 0:
            raise RuntimeError(f"Found 0 images in {self.category / self._SPLIT}")

    def _download(self) -> None:
        """
        Download the Anomaly dataset from URL
        """
        if (self.root / self.category).is_dir():
            logger.warning("Dataset directory exists.")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            self.filename = self.root / "anomaly_dataset.tar.xz"

            if self.download_url is None:
                raise RuntimeError("Please specify url to download dataset")

            logger.info("Downloading Anomaly Dataset")
            with DownloadProgressBar(
                    unit="B", unit_scale=True, miniters=1, desc=self.download_url.split("/")[-1]
            ) as progress_bar:
                urlretrieve(url=self.download_url, filename=self.filename, reporthook=progress_bar.update_to)

            self._extract()
            self._clean()

    def _extract(self) -> None:
        """
        Extract Anomaly Dataset
        """
        logger.info("Extracting Anomaly dataset")
        with tarfile.open(self.filename) as file:
            file.extractall(self.root)

    def _clean(self) -> None:
        """
        Cleanup Anomaly Dataset tar file.
        """
        logger.info("Cleaning up the tar file")
        self.filename.unlink()

    def __len__(self) -> int:
        return len(self.samples)

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError()


class AnomalyTrainDS(BaseAnomalyDataset):
    """
    Anomaly Training dataset
    """

    _SPLIT = "train"

    def __getitem__(self, index: int) -> Union[Dict[str, Tensor], Dict[str, Union[str, Tensor]]]:
        image_path = self.samples.image_path[index]

        image = read_image(image_path)
        augmented = self.transform(image=image)
        image_tensor = augmented["image"]

        sample = {"image": image_tensor}

        return sample


class AnomalyTestClassificationDS(BaseAnomalyDataset):
    """
    Anomaly classification - test dataset
    """

    _SPLIT = "test"

    def __getitem__(self, index: int) -> Union[Dict[str, Tensor], Dict[str, Union[str, Tensor]]]:
        image_path = self.samples.image_path[index]
        label_index = self.samples.label_index[index]

        image = read_image(image_path)
        augmented = self.transform(image=image)
        image_tensor = augmented["image"]

        sample = {
            "image_path": image_path,
            "image": image_tensor,
            "label": label_index,
        }

        return sample


class AnomalyTestSegmentationDS(BaseAnomalyDataset):
    """
    Anomaly segmentation - test dataset
    """

    _TARGET_FILE_EXT = "_mask.png"
    _SPLIT = "test"

    def __getitem__(self, index: int) -> Union[Dict[str, Tensor], Dict[str, Union[str, Tensor]]]:
        image_path = self.samples.image_path[index]
        mask_path = self.samples.target_path[index]
        label_index = self.samples.label_index[index]

        image = read_image(image_path)
        if label_index == 0:
            mask = np.zeros((image.shape[0], image.shape[1]))
        else:
            mask = cv2.imread((mask_path + self._TARGET_FILE_EXT), 0) / 255.0

        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented["image"]
        mask_tensor = augmented["mask"]

        sample = {
            "image_path": image_path,
            "mask_path": mask_path,
            "image": image_tensor,
            "label": label_index,
            "mask": mask_tensor,
        }

        return sample


class AnomalyTestDetectionDS(BaseAnomalyDataset):
    """
    Anomaly detection - test dataset
    """

    _TARGET_FILE_EXT = ".xml"
    _SPLIT = "test"

    def __init__(
        self,
        root: Union[Path, str],
        category: Path,
        label_format: str = None,
        transform_params: Union[DictConfig, ListConfig] = None,
        download: bool = False,
        download_url: Optional[str] = None,
        samples: Optional[DataFrame] = None,
    ) -> None:
        super().__init__(root, category, label_format, transform_params, download, download_url, samples)

        if self.label_format == "pascal_voc":
            self.label_parser = PascalVocReader
        else:
            raise ValueError(f"Unknown data annotation format: {self.label_format}!")

    def __getitem__(self, index: int) -> Union[Dict[str, Tensor], Dict[str, Union[str, Tensor]]]:
        image_path = self.samples.image_path[index]
        target_path = self.samples.target_path[index]
        label_index = self.samples.label_index[index]

        image = read_image(image_path)
        gt_bbox: Dict[str, Any] = {}
        if label_index != 0 and self.label_parser is not None:
            gt_bbox = self.label_parser(target_path + self._TARGET_FILE_EXT).get_shapes()

        augmented = self.transform(image=image, bbox=gt_bbox)
        image_tensor = augmented["image"]
        bbox_t = augmented["bbox"]

        sample = {
            "image_path": image_path,
            "target_path": target_path,
            "image": image_tensor,
            "label": label_index,
            "bbox_t": bbox_t,
        }

        return sample


class AnomalyDataModule(LightningDataModule):
    """
    Anomaly data Lightning Module

    init parameters:
        root: folder containing the dataset
        url: web link to download a dataset
        category: subcategory or class label
        task: specifies usage of dataset to either classification, detection or segmentation
        label_format: data format for annotations/labels
        batch_size: number of images per batch
        num_workers: number of parallel thread to be used for data loading
        transform_params: parameters for image transforms
    """

    def __init__(
        self,
        root: Union[Path, str],
        url: str,
        category: str,
        task: str,
        label_format: str,
        train_batch_size: int,
        test_batch_size: int,
        num_workers: int,
        transform_params: Union[DictConfig, ListConfig],
    ) -> None:
        super().__init__()

        self.root = root if isinstance(root, Path) else Path(root)
        self.url = str(url)
        self.category = Path(category)
        self.task = task
        self.label_format = label_format
        self.dataset_path = self.root / self.category
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.transform_params = transform_params
        self.train_dataset = AnomalyTrainDS

        self.train_data: BaseAnomalyDataset
        self.test_data: BaseAnomalyDataset

        if self.task == "detection":
            self.test_dataset = AnomalyTestDetectionDS
        elif self.task == "classification":
            self.test_dataset = AnomalyTestClassificationDS
        elif self.task == "segmentation":
            self.test_dataset = AnomalyTestSegmentationDS
        else:
            raise ValueError(f"Unknown task type: {self.task}!")

    def prepare_data(self):
        """
        prepare_data
            download training data if not available
        """

        # Training Data
        self.train_dataset(
            root=self.root,
            category=self.category,
            label_format=self.label_format,
            transform_params=self.transform_params,
            download=True,
            download_url=self.url,
        )
        # Test Data
        self.test_dataset(
            root=self.root,
            category=self.category,
            label_format=self.label_format,
            transform_params=self.transform_params,
            download=True,
            download_url=self.url,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        setup:
            Data preparation - split for train, test & val

        Args:
            stage: optinal argument to specify if train or test
        """
        samples_df = make_dataset(path=self.root / self.category)

        # Get the data frame for the train dataset.
        samples_train = samples_df[samples_df.split == "train"]
        samples_train = samples_train.reset_index(drop=True)
        self.train_data = self.train_dataset(
            root=self.root,
            category=self.category,
            label_format=self.label_format,
            transform_params=self.transform_params,
            samples=samples_train,
        )

        # Get the data frame for the test dataset.
        samples_test = samples_df[samples_df.split == "test"]
        samples_test = samples_test.reset_index(drop=True)
        self.test_data = self.test_dataset(
            root=self.root,
            category=self.category,
            label_format=self.label_format,
            transform_params=self.transform_params,
            samples=samples_test,
        )

    def train_dataloader(self) -> DataLoader:
        """
        Train Dataloader
        """
        return DataLoader(
            self.train_data, shuffle=False, batch_size=self.train_batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        """
        Validation Dataloader
        """
        return DataLoader(self.test_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """
        Test Dataloader
        """
        return DataLoader(self.test_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers)
