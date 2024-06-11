"""MVTec AD Dataset (CC BY-NC-SA 4.0).

Description:
    This script contains PyTorch Dataset, Dataloader and PyTorch Lightning
    DataModule for the MVTec AD dataset. If the dataset is not on the file system,
    the script downloads and extracts the dataset and create PyTorch data objects.

License:
    MVTec AD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).

References:
    - Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger:
      The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for
      Unsupervised Anomaly Detection; in: International Journal of Computer Vision
      129(4):1038-1059, 2021, DOI: 10.1007/s11263-020-01400-4.

    - Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: MVTec AD —
      A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection;
      in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
      9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982.
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from pathlib import Path

from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import (
    DownloadInfo,
    LabelName,
    Split,
    TestSplitMode,
    ValSplitMode,
    download_and_extract,
    validate_path,
)

logger = logging.getLogger(__name__)


IMG_EXTENSIONS = (".png", ".PNG")

DOWNLOAD_INFO = DownloadInfo(
    name="mvtec",
    url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094"
    "/mvtec_anomaly_detection.tar.xz",
    hashsum="cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d",
)

CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


def make_mvtec_dataset(
    root: str | Path,
    split: str | Split | None = None,
    extensions: Sequence[str] | None = None,
) -> DataFrame:
    """Create MVTec AD samples by parsing the MVTec AD data file structure.

    The files are expected to follow the structure:
        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/mask_filename.png

    This function creates a dataframe to store the parsed information based on the following format:

    +---+---------------+-------+---------+---------------+---------------------------------------+-------------+
    |   | path          | split | label   | image_path    | mask_path                             | label_index |
    +===+===============+=======+=========+===============+=======================================+=============+
    | 0 | datasets/name | test  | defect  | filename.png  | ground_truth/defect/filename_mask.png | 1           |
    +---+---------------+-------+---------+---------------+---------------------------------------+-------------+

    Args:
        root (Path): Path to dataset
        split (str | Split | None, optional): Dataset split (ie., either train or test).
            Defaults to ``None``.
        extensions (Sequence[str] | None, optional): List of file extensions to be included in the dataset.
            Defaults to ``None``.

    Examples:
        The following example shows how to get training samples from MVTec AD bottle category:

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
        DataFrame: an output dataframe containing the samples of the dataset.
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = validate_path(root)
    samples_list = [(str(root),) + f.parts[-3:] for f in root.glob(r"**/*") if f.suffix in extensions]
    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    samples = DataFrame(samples_list, columns=["path", "split", "label", "image_path"])

    # Modify image_path column by converting to absolute path
    samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    # separate masks from samples
    mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(by="image_path", ignore_index=True)
    samples = samples[samples.split != "ground_truth"].sort_values(by="image_path", ignore_index=True)

    # assign mask paths to anomalous test images
    samples["mask_path"] = ""
    samples.loc[
        (samples.split == "test") & (samples.label_index == LabelName.ABNORMAL),
        "mask_path",
    ] = mask_samples.image_path.to_numpy()

    # assert that the right mask files are associated with the right test images
    abnormal_samples = samples.loc[samples.label_index == LabelName.ABNORMAL]
    if (
        len(abnormal_samples)
        and not abnormal_samples.apply(lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1).all()
    ):
        msg = """Mismatch between anomalous images and ground truth masks. Make sure t
        he mask files in 'ground_truth' folder follow the same naming convention as the
        anomalous images in the dataset (e.g. image: '000.png', mask: '000.png' or '000_mask.png')."""
        raise MisMatchError(msg)

    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples


class MVTecDataset(AnomalibDataset):
    """MVTec dataset class.

    Args:
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``.
        root (Path | str): Path to the root of the dataset.
            Defaults to ``./datasets/MVTec``.
        category (str): Sub-category of the dataset, e.g. 'bottle'
            Defaults to ``bottle``.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST
            Defaults to ``None``.

    Examples:
        .. code-block:: python

            from anomalib.data.image.mvtec import MVTecDataset
            from anomalib.data.utils.transforms import get_transforms

            transform = get_transforms(image_size=256)
            dataset = MVTecDataset(
                task="classification",
                transform=transform,
                root='./datasets/MVTec',
                category='zipper',
            )
            dataset.setup()
            print(dataset[0].keys())
            # Output: dict_keys(['image_path', 'label', 'image'])

        When the task is segmentation, the dataset will also contain the mask:

        .. code-block:: python

            dataset.task = "segmentation"
            dataset.setup()
            print(dataset[0].keys())
            # Output: dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

        The image is a torch tensor of shape (C, H, W) and the mask is a torch tensor of shape (H, W).

        .. code-block:: python

            print(dataset[0]["image"].shape, dataset[0]["mask"].shape)
            # Output: (torch.Size([3, 256, 256]), torch.Size([256, 256]))

    """

    def __init__(
        self,
        task: TaskType,
        root: Path | str = "./datasets/MVTec",
        category: str = "bottle",
        transform: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(task=task, transform=transform)

        self.root_category = Path(root) / Path(category)
        self.category = category
        self.split = split
        self.samples = make_mvtec_dataset(self.root_category, split=self.split, extensions=IMG_EXTENSIONS)


class MVTec(AnomalibDataModule):
    """MVTec Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/MVTec"``.
        category (str): Category of the MVTec dataset (e.g. "bottle" or "cable").
            Defaults to ``"bottle"``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        task TaskType): Task type, 'classification', 'detection' or 'segmentation'
            Defaults to ``TaskType.SEGMENTATION``.
        image_size (tuple[int, int], optional): Size to which input images should be resized.
            Defaults to ``None``.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        train_transform (Transform, optional): Transforms that should be applied to the input images during training.
            Defaults to ``None``.
        eval_transform (Transform, optional): Transforms that should be applied to the input images during evaluation.
            Defaults to ``None``.
        test_split_mode (TestSplitMode): Setting that determines how the testing subset is obtained.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of images from the train set that will be reserved for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
            Defualts to ``None``.

    Examples:
        To create an MVTec AD datamodule with default settings:

        >>> datamodule = MVTec()
        >>> datamodule.setup()
        >>> i, data = next(enumerate(datamodule.train_dataloader()))
        >>> data.keys()
        dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

        >>> data["image"].shape
        torch.Size([32, 3, 256, 256])

        To change the category of the dataset:

        >>> datamodule = MVTec(category="cable")

        To change the image and batch size:

        >>> datamodule = MVTec(image_size=(512, 512), train_batch_size=16, eval_batch_size=8)

        MVTec AD dataset does not provide a validation set. If you would like
        to use a separate validation set, you can use the ``val_split_mode`` and
        ``val_split_ratio`` arguments to create a validation set.

        >>> datamodule = MVTec(val_split_mode=ValSplitMode.FROM_TEST, val_split_ratio=0.1)

        This will subsample the test set by 10% and use it as the validation set.
        If you would like to create a validation set synthetically that would
        not change the test set, you can use the ``ValSplitMode.SYNTHETIC`` option.

        >>> datamodule = MVTec(val_split_mode=ValSplitMode.SYNTHETIC, val_split_ratio=0.2)

    """

    def __init__(
        self,
        root: Path | str = "./datasets/MVTec",
        category: str = "bottle",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType | str = TaskType.SEGMENTATION,
        image_size: tuple[int, int] | None = None,
        transform: Transform | None = None,
        train_transform: Transform | None = None,
        eval_transform: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            image_size=image_size,
            transform=transform,
            train_transform=train_transform,
            eval_transform=eval_transform,
            num_workers=num_workers,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.task = TaskType(task)
        self.root = Path(root)
        self.category = category

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.

        Note:
            The stage argument is not used here. This is because, for a given instance of an AnomalibDataModule
            subclass, all three subsets are created at the first call of setup(). This is to accommodate the subset
            splitting behaviour of anomaly tasks, where the validation set is usually extracted from the test set, and
            the test set must therefore be created as early as the `fit` stage.

        """
        self.train_data = MVTecDataset(
            task=self.task,
            transform=self.train_transform,
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
        )
        self.test_data = MVTecDataset(
            task=self.task,
            transform=self.eval_transform,
            split=Split.TEST,
            root=self.root,
            category=self.category,
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

                >> datamodule = MVTec(root="./datasets/MVTec", category="bottle")
                >> datamodule.prepare_data()

            After:

            .. code-block:: bash

                $ tree datasets
                datasets
                ├── dataset1
                ├── dataset2
                └── MVTec
                    ├── bottle
                    ├── ...
                    └── zipper
        """
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)
