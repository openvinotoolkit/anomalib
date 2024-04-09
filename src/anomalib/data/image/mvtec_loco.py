"""MVTec LOCO AD Dataset (CC BY-NC-SA 4.0).

Description:
    This script contains PyTorch Dataset, Dataloader and PyTorch Lightning
    DataModule for the MVTec LOCO AD dataset. If the dataset is not on the file system,
    the script downloads and extracts the dataset and create PyTorch data objects.

License:
    MVTec LOCO AD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).

References:
    - Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, and Carsten Steger:
      Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization;
      in: International Journal of Computer Vision (IJCV) 130, 947-969, 2022, DOI: 10.1007/s11263-022-01578-9
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from pandas import DataFrame
from torchvision.transforms.v2 import Transform
from torchvision.tv_tensors import Mask

from anomalib import TaskType
from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.utils import (
    DownloadInfo,
    LabelName,
    Split,
    TestSplitMode,
    ValSplitMode,
    download_and_extract,
    masks_to_boxes,
    read_image,
    read_mask,
    validate_path,
)

logger = logging.getLogger(__name__)


IMG_EXTENSIONS = (".png", ".PNG")

DOWNLOAD_INFO = DownloadInfo(
    name="mvtec_loco",
    url="https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701"
    "/mvtec_loco_anomaly_detection.tar.xz",
    hashsum="9e7c84dba550fd2e59d8e9e231c929c45ba737b6b6a6d3814100f54d63aae687",
)

CATEGORIES = (
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
)


def make_mvtec_loco_dataset(
    root: str | Path,
    split: str | Split | None = None,
    extensions: Sequence[str] = IMG_EXTENSIONS,
) -> DataFrame:
    """Create MVTec LOCO AD samples by parsing the original MVTec LOCO AD data file structure.

    The files are expected to follow the structure:
        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/image_filename/000.png

    where there can be multiple ground-truth masks for the corresponding anomalous images.

    This function creates a dataframe to store the parsed information based on the following format:

    +---+---------------+-------+---------+-------------------------+-----------------------------+-------------+
    |   | path          | split | label   | image_path              | mask_path                  | label_index |
    +===+===============+=======+=========+===============+=======================================+=============+
    | 0 | datasets/name | test  | defect  | path/to/image/file.png  | [path/to/masks/file.png]    | 1           |
    +---+---------------+-------+---------+-------------------------+-----------------------------+-------------+

    Args:
        root (str | Path): Path to dataset
        split (str | Split | None): Dataset split (ie., either train or test).
            Defaults to ``None``.
        extensions (Sequence[str]): List of file extensions to be included in the dataset.
            Defaults to ``None``.

    Returns:
        DataFrame: an output dataframe containing the samples of the dataset.

    Examples:
        The following example shows how to get test samples from MVTec LOCO AD pushpins category:

        >>> root = Path('./MVTec_LOCO')
        >>> category = 'pushpins'
        >>> path = root / category
        >>> samples = make_mvtec_loco_dataset(path, split='test')
    """
    root = validate_path(root)

    # Retrieve the image and mask files
    samples_list = []
    for f in root.glob("**/*"):
        if f.suffix in extensions:
            parts = f.parts
            # 'ground_truth' and non 'ground_truth' path have a different structure
            if "ground_truth" not in parts:
                split_folder, label_folder, image_file = parts[-3:]
                image_path = f"{root}/{split_folder}/{label_folder}/{image_file}"
                samples_list.append((str(root), split_folder, label_folder, "", image_path))
            else:
                split_folder, label_folder, image_folder, image_file = parts[-4:]
                image_path = f"{root}/{split_folder}/{label_folder}/{image_folder}/{image_file}"
                samples_list.append((str(root), split_folder, label_folder, image_folder, image_path))

    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    samples = DataFrame(samples_list, columns=["path", "split", "label", "image_folder", "image_path"])

    # Replace validation to Split.VAL.value in the split column
    samples["split"] = samples["split"].replace("validation", Split.VAL.value)

    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    # separate ground-truth masks from samples
    mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(by="image_path", ignore_index=True)
    samples = samples[samples.split != "ground_truth"].sort_values(by="image_path", ignore_index=True)

    # Group masks and aggregate the path into a list
    mask_samples = (
        mask_samples.groupby(["path", "split", "label", "image_folder"])["image_path"]
        .agg(list)
        .reset_index()
        .rename(columns={"image_path": "mask_path"})
    )

    # assign mask paths to anomalous test images
    samples["mask_path"] = ""
    samples.loc[
        (samples.split == "test") & (samples.label_index == LabelName.ABNORMAL),
        "mask_path",
    ] = mask_samples.mask_path.to_numpy()

    # validate that the right mask files are associated with the right test images
    if len(samples.loc[samples.label_index == LabelName.ABNORMAL]):
        image_stems = samples.loc[samples.label_index == LabelName.ABNORMAL]["image_path"].apply(lambda x: Path(x).stem)
        mask_parent_stems = samples.loc[samples.label_index == LabelName.ABNORMAL]["mask_path"].apply(
            lambda x: {Path(mask_path).parent.stem for mask_path in x},
        )

        if not all(
            next(iter(mask_stems)) == image_stem
            for image_stem, mask_stems in zip(image_stems, mask_parent_stems, strict=True)
        ):
            error_message = (
                "Mismatch between anomalous images and ground truth masks. "
                "Make sure the parent folder of the mask files in 'ground_truth' folder "
                "follows the same naming convention as the anomalous images in the dataset "
                "(e.g., image: '005.png', mask: '005/000.png')."
            )
            raise ValueError(error_message)

    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples


class MVTecLocoDataset(AnomalibDataset):
    """MVTec LOCO dataset class.

    Args:
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``.
        root (Path | str): Path to the root of the dataset.
            Defaults to ``./datasets/MVTec_LOCO``.
        category (str): Sub-category of the dataset, e.g. 'breakfast_box'
            Defaults to ``breakfast_box``.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None): Split of the dataset, Split.TRAIN, Split.VAL, or Split.TEST
            Defaults to ``None``.

    Examples:
        .. code-block:: python

            from anomalib.data.image.mvtec_loco import MVTecLocoDataset
            from anomalib.data.utils.transforms import get_transforms
            from torchvision.transforms.v2 import Resize

            transform = Resize((256, 256))
            dataset = MVTecLocoDataset(
                task="classification",
                transform=transform,
                root='./datasets/MVTec_LOCO',
                category='breakfast_box',
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
        root: Path | str = "./datasets/MVTec_LOCO",
        category: str = "breakfast_box",
        transform: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(task=task, transform=transform)

        self.root_category = Path(root) / category
        self.split = split
        self.samples = make_mvtec_loco_dataset(
            self.root_category,
            split=self.split,
            extensions=IMG_EXTENSIONS,
        )

    def __getitem__(self, index: int) -> dict[str, str | torch.Tensor]:
        """Get dataset item for the index ``index``.

        This method is mostly based on the super class implementation, with some different as follows:
            - Using 'torch.where' to make sure the 'mask' in the return item is binarized
            - An additional 'masks' is added, the non-binary masks with original size for the SPRO metric calculation
        Args:
            index (int): Index to get the item.

        Returns:
            dict[str, str | torch.Tensor]: Dict of image tensor during training. Otherwise, Dict containing image path,
                target path, image tensor, label and transformed bounding box.
        """
        image_path = self.samples.iloc[index].image_path
        mask_path = self.samples.iloc[index].mask_path
        label_index = self.samples.iloc[index].label_index

        image = read_image(image_path, as_tensor=True)
        item = {"image_path": image_path, "label": label_index}

        if self.task == TaskType.CLASSIFICATION:
            item["image"] = self.transform(image) if self.transform else image
        elif self.task in (TaskType.DETECTION, TaskType.SEGMENTATION):
            # Only Anomalous (1) images have masks in anomaly datasets
            # Therefore, create empty mask for Normal (0) images.
            if isinstance(mask_path, str):
                mask_path = [mask_path]
            masks = (
                Mask(torch.zeros(image.shape[-2:])).to(torch.uint8)
                if label_index == LabelName.NORMAL
                else Mask(torch.stack([read_mask(path, as_tensor=True) for path in mask_path]))
            )
            mask = Mask(masks.view(-1, *masks.shape[-2:]).any(dim=0).to(torch.uint8))
            item["image"], item["mask"] = self.transform(image, mask) if self.transform else (image, mask)

            item["mask_path"] = mask_path
            # List of masks with the original size for saturation based metrics calculation
            item["masks"] = masks

            if self.task == TaskType.DETECTION:
                # create boxes from masks for detection task
                boxes, _ = masks_to_boxes(item["mask"])
                item["boxes"] = boxes[0]
        else:
            msg = f"Unknown task type: {self.task}"
            raise ValueError(msg)

        return item


class MVTecLoco(AnomalibDataModule):
    """MVTec LOCO Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/MVTec_LOCO"``.
        category (str): Category of the MVTec LOCO dataset (e.g. "breakfast_box").
            Defaults to ``"breakfast_box"``.
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
            Defaults to ``ValSplitMode.FROM_DIR``.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
            Defaults to ``None``.

    Examples:
        To create an MVTec LOCO AD datamodule with default settings:

        >>> datamodule = MVTecLoco(root="anomalib/datasets/MVTec_LOCO")
        >>> datamodule.setup()
        >>> i, data = next(enumerate(datamodule.train_dataloader()))
        >>> data.keys()
        dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

        >>> data["image"].shape
        torch.Size([32, 3, 256, 256])

        To change the category of the dataset:

        >>> datamodule = MVTecLoco(category="pushpins")

        To change the image and batch size:

        >>> datamodule = MVTecLoco(image_size=(512, 512), train_batch_size=16, eval_batch_size=8)

        MVTec LOCO AD dataset provide an independent validation set with normal images only in the 'validation' folder.
        If you would like to use a different validation set splitted from train or test set,
        you can use the ``val_split_mode`` and ``val_split_ratio`` arguments to create a new validation set.

        >>> datamodule = MVTecLoco(val_split_mode=ValSplitMode.FROM_TEST, val_split_ratio=0.1)

        This will subsample the test set by 10% and use it as the validation set.
        If you would like to create a validation set synthetically that would
        not change the test set, you can use the ``ValSplitMode.SYNTHETIC`` option.

        >>> datamodule = MVTecLoco(val_split_mode=ValSplitMode.SYNTHETIC, val_split_ratio=0.2)
    """

    def __init__(
        self,
        root: Path | str = "./datasets/MVTec_LOCO",
        category: str = "breakfast_box",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType = TaskType.SEGMENTATION,
        image_size: tuple[int, int] | None = None,
        transform: Transform | None = None,
        train_transform: Transform | None = None,
        eval_transform: Transform | None = None,
        test_split_mode: TestSplitMode = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode = ValSplitMode.FROM_DIR,
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
        self.task = task
        self.root = Path(root)
        self.category = category

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets, configs, and perform dynamic subset splitting.

        This method overrides the parent class's method to also setup the val dataset.
        The MVTec LOCO dataset provides an independent validation subset.
        """
        self.train_data = MVTecLocoDataset(
            task=self.task,
            transform=self.train_transform,
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
        )
        self.val_data = MVTecLocoDataset(
            task=self.task,
            transform=self.eval_transform,
            split=Split.VAL,
            root=self.root,
            category=self.category,
        )
        self.test_data = MVTecLocoDataset(
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

                >> datamodule = MVTecLoco(root="./datasets/MVTec_LOCO", category="breakfast_box")
                >> datamodule.prepare_data()

            After:

            .. code-block:: bash

                $ tree datasets
                datasets
                ├── dataset1
                ├── dataset2
                └── MVTec_LOCO
                    ├── breakfast_box
                    ├── ...
                    └── splicing_connectors
        """
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)
