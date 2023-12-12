"""BTech Dataset.

This script contains PyTorch Lightning DataModule for the BTech dataset.

If the dataset is not on the file system, the script downloads and
extracts the dataset and create PyTorch data objects.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import shutil
from pathlib import Path

import albumentations as A  # noqa: N812
import cv2
import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import tqdm

from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.utils import (
    DownloadInfo,
    InputNormalizationMethod,
    LabelName,
    Split,
    TestSplitMode,
    ValSplitMode,
    download_and_extract,
    get_transforms,
)
from anomalib.utils.types import TaskType

logger = logging.getLogger(__name__)

DOWNLOAD_INFO = DownloadInfo(
    name="btech",
    url="https://avires.dimi.uniud.it/papers/btad/btad.zip",
    checksum="c1fa4d56ac50dd50908ce04e81037a8e",
)

CATEGORIES = ("01", "02", "03")


def make_btech_dataset(path: Path, split: str | Split | None = None) -> DataFrame:
    """Create BTech samples by parsing the BTech data file structure.

    The files are expected to follow the structure:

        .. code-block:: bash

            path/to/dataset/split/category/image_filename.png
            path/to/dataset/ground_truth/category/mask_filename.png

    Args:
        path (Path): Path to dataset
        split (str | Split | None, optional): Dataset split (ie., either train or test).
            Defaults to ``None``.

    Example:
        The following example shows how to get training samples from BTech 01 category:

        .. code-block:: python

            >>> root = Path('./BTech')
            >>> category = '01'
            >>> path = root / category
            >>> path
            PosixPath('BTech/01')

            >>> samples = make_btech_dataset(path, split='train')
            >>> samples.head()
            path     split label image_path                  mask_path                     label_index
            0  BTech/01 train 01    BTech/01/train/ok/105.bmp BTech/01/ground_truth/ok/105.png      0
            1  BTech/01 train 01    BTech/01/train/ok/017.bmp BTech/01/ground_truth/ok/017.png      0
            ...

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test)
    """
    samples_list = [
        (str(path),) + filename.parts[-3:] for filename in path.glob("**/*") if filename.suffix in (".bmp", ".png")
    ]
    if not samples_list:
        msg = f"Found 0 images in {path}"
        raise RuntimeError(msg)

    samples = pd.DataFrame(samples_list, columns=["path", "split", "label", "image_path"])
    samples = samples[samples.split != "ground_truth"]

    # Create mask_path column
    # (safely handles cases where non-mask image_paths end with either .png or .bmp)
    samples["mask_path"] = (
        samples.path
        + "/ground_truth/"
        + samples.label
        + "/"
        + samples.image_path.str.rstrip("png").str.rstrip(".").str.rstrip("bmp").str.rstrip(".")
        + ".png"
    )

    # Modify image_path column by converting to absolute path
    samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    # Good images don't have mask
    samples.loc[(samples.split == "test") & (samples.label == "ok"), "mask_path"] = ""

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "ok"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "ok"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    # Get the data frame for the split.
    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples


class BTechDataset(AnomalibDataset):
    """Btech Dataset class.

    Args:
        root: Path to the BTech dataset
        category: Name of the BTech category.
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
        split: 'train', 'val' or 'test'
        task: ``classification``, ``detection`` or ``segmentation``
        create_validation_set: Create a validation subset in addition to the train and test subsets

    Examples:
        >>> from anomalib.data.image.btech import BTechDataset
        >>> from anomalib.data.utils.transforms import get_transforms
        >>> transform = get_transforms(image_size=256)
        >>> dataset = BTechDataset(
        ...     task="classification",
        ...     transform=transform,
        ...     root='./datasets/BTech',
        ...     category='01',
        ... )
        >>> dataset[0].keys()
        >>> dataset.setup()
        dict_keys(['image'])

        >>> dataset.split = "test"
        >>> dataset[0].keys()
        dict_keys(['image', 'image_path', 'label'])

        >>> dataset.task = "segmentation"
        >>> dataset.split = "train"
        >>> dataset[0].keys()
        dict_keys(['image'])

        >>> dataset.split = "test"
        >>> dataset[0].keys()
        dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])

        >>> dataset[0]["image"].shape, dataset[0]["mask"].shape
        (torch.Size([3, 256, 256]), torch.Size([256, 256]))
    """

    def __init__(
        self,
        root: str | Path,
        category: str,
        transform: A.Compose,
        split: str | Split | None = None,
        task: TaskType = TaskType.SEGMENTATION,
    ) -> None:
        super().__init__(task, transform)

        self.root_category = Path(root) / category
        self.split = split

    def _setup(self) -> None:
        self.samples = make_btech_dataset(path=self.root_category, split=self.split)


class BTech(AnomalibDataModule):
    """BTech Lightning Data Module.

    Args:
        root (Path | str): Path to the BTech dataset.
            Defaults to ``"./datasets/BTech"``.
        category (str): Name of the BTech category.
            Defaults to ``"01"``.
        image_size (int | tuple[int, int] | None, optional): Variable to which image is resized.
            Defaults to ``(256, 256)``.
        center_crop (int | tuple[int, int] | None, optional): When provided, the images will be center-cropped to the
            provided dimensions.
            Defaults to ``None``.
        normalization (str | InputNormalizationMethod, optional): When True, the images will be normalized to the
            ImageNet statistics.
            Defaults to ``InputNormalizationMethod.IMAGENET``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Eval batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        task (TaskType, optional): Task type.
            Defaults to ``TaskType.SEGMENTATION``.
        transform_config_train (str | A.Compose | None, optional): Config for pre-processing during training.
            Defaults to ``None``.
        transform_config_eval (str | A.Compose | None, optional): Config for pre-processing during validation.
            Defaults to ``None``.
        test_split_mode (TestSplitMode, optional): Setting that determines how the testing subset is obtained.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float, optional): Fraction of images from the train set that will be reserved for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode, optional): Setting that determines how the validation subset is obtained.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float, optional): Fraction of train or test images that will be reserved for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
            Defaults to ``None``.

    Examples:
        To create the BTech datamodule, we need to instantiate the class, and call the ``setup`` method.

        >>> from anomalib.data import BTech
        >>> datamodule = BTech(
        ...     root="./datasets/BTech",
        ...     category="01",
        ...     image_size=256,
        ...     train_batch_size=32,
        ...     eval_batch_size=32,
        ...     num_workers=8,
        ...     transform_config_train=None,
        ...     transform_config_eval=None,
        ... )
        >>> datamodule.setup()

        To get the train dataloader and the first batch of data:

        >>> i, data = next(enumerate(datamodule.train_dataloader()))
        >>> data.keys()
        dict_keys(['image'])
        >>> data["image"].shape
        torch.Size([32, 3, 256, 256])

        To access the validation dataloader and the first batch of data:

        >>> i, data = next(enumerate(datamodule.val_dataloader()))
        >>> data.keys()
        dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])
        >>> data["image"].shape, data["mask"].shape
        (torch.Size([32, 3, 256, 256]), torch.Size([32, 256, 256]))

        Similarly, to access the test dataloader and the first batch of data:

        >>> i, data = next(enumerate(datamodule.test_dataloader()))
        >>> data.keys()
        dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])
        >>> data["image"].shape, data["mask"].shape
        (torch.Size([32, 3, 256, 256]), torch.Size([32, 256, 256]))
    """

    def __init__(
        self,
        root: Path | str = "./datasets/BTech",
        category: str = "01",
        image_size: int | tuple[int, int] = (256, 256),
        center_crop: int | tuple[int, int] | None = None,
        normalization: str | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType = TaskType.SEGMENTATION,
        transform_config_train: str | A.Compose | None = None,
        transform_config_eval: str | A.Compose | None = None,
        test_split_mode: TestSplitMode = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.root = Path(root)
        self.category = Path(category)

        transform_train = get_transforms(
            config=transform_config_train,
            image_size=image_size,
            center_crop=center_crop,
            normalization=InputNormalizationMethod(normalization),
        )
        transform_eval = get_transforms(
            config=transform_config_eval,
            image_size=image_size,
            center_crop=center_crop,
            normalization=InputNormalizationMethod(normalization),
        )

        self.train_data = BTechDataset(
            task=task,
            transform=transform_train,
            split=Split.TRAIN,
            root=root,
            category=category,
        )
        self.test_data = BTechDataset(
            task=task,
            transform=transform_eval,
            split=Split.TEST,
            root=root,
            category=category,
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available.

        This method checks if the specified dataset is available in the file system.
        If not, it downloads and extracts the dataset into the appropriate directory.

        Example:
            Assume the dataset is not available on the file system.
            Here's how the directory structure looks before and after calling the
            `prepare_data` method:

            Before:

            .. code-block:: bash

                $ tree datasets
                datasets
                ├── dataset1
                └── dataset2

            Calling the method:

            .. code-block:: python

                >> datamodule = BTech(root="./datasets/BTech", category="01")
                >> datamodule.prepare_data()

            After:

            .. code-block:: bash

                $ tree datasets
                datasets
                ├── dataset1
                ├── dataset2
                └── BTech
                    ├── 01
                    ├── 02
                    └── 03
        """
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root.parent, DOWNLOAD_INFO)

            # rename folder and convert images
            logger.info("Renaming the dataset directory")
            shutil.move(src=str(self.root.parent / "BTech_Dataset_transformed"), dst=str(self.root))
            logger.info("Convert the bmp formats to png to have consistent image extensions")
            for filename in tqdm(self.root.glob("**/*.bmp"), desc="Converting bmp to png"):
                image = cv2.imread(str(filename))
                cv2.imwrite(str(filename.with_suffix(".png")), image)
                filename.unlink()
