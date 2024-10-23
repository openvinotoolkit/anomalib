"""Custom Folder Data Module.

This script creates a custom Lightning DataModule from a folder.
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.folder import FolderDataset
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode


class Folder(AnomalibDataModule):
    """Folder DataModule.

    Args:
        name (str): Name of the dataset. This is used to name the datamodule, especially when logging/saving.
        normal_dir (str | Path | Sequence[str | Path]): Path to the directory containing normal images.
        root (str | Path | None): Path to the root folder containing normal and abnormal dirs. Defaults to ``None``.
        abnormal_dir (str | Path | Sequence[str | Path] | None): Path to the directory containing abnormal images.
            Defaults to ``None``.
        normal_test_dir (str | Path | Sequence[str | Path] | None): Path to the directory containing
            normal images for the test dataset. Defaults to ``None``.
        mask_dir (str | Path | Sequence[str | Path] | None): Path to the directory containing
            the mask annotations. Defaults to ``None``.
        normal_split_ratio (float): Ratio to split normal training images and add to the
            test set in case test set doesn't contain any normal images. Defaults to 0.2.
        extensions (tuple[str, ...] | None): Type of the image extensions to read from the
            directory. Defaults to ``None``.
        train_batch_size (int): Training batch size. Defaults to 32.
        eval_batch_size (int): Validation, test and predict batch size. Defaults to 32.
        num_workers (int): Number of workers. Defaults to 8.
        task (TaskType | str): Task type. Could be ``classification``, ``detection`` or ``segmentation``.
            Defaults to ``TaskType.SEGMENTATION``.
        image_size (tuple[int, int] | None): Size to which input images should be resized. Defaults to ``None``.
        transform (Transform | None): Transforms that should be applied to the input images. Defaults to ``None``.
        train_transform (Transform | None): Transforms that should be applied to the input images during training.
            Defaults to ``None``.
        eval_transform (Transform | None): Transforms that should be applied to the input images during evaluation.
            Defaults to ``None``.
        test_split_mode (TestSplitMode | str): Setting that determines how the testing subset is obtained.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of images from the train set that will be reserved for testing.
            Defaults to 0.2.
        val_split_mode (ValSplitMode | str): Setting that determines how the validation subset is obtained.
            Defaults to ``ValSplitMode.FROM_TEST``.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
            Defaults to 0.5.
        seed (int | None): Seed used during random subset splitting. Defaults to ``None``.

    Examples:
        The following code demonstrates how to use the ``Folder`` datamodule:

        >>> from pathlib import Path
        >>> from anomalib.data import Folder
        >>> from anomalib import TaskType

        >>> dataset_root = Path("./sample_dataset")
        >>> folder_datamodule = Folder(
        ...     name="my_folder_dataset",
        ...     root=dataset_root,
        ...     normal_dir="good",
        ...     abnormal_dir="crack",
        ...     task=TaskType.SEGMENTATION,
        ...     mask_dir=dataset_root / "mask" / "crack",
        ...     image_size=(256, 256),
        ...     train_batch_size=32,
        ...     eval_batch_size=32,
        ...     num_workers=8,
        ... )
        >>> folder_datamodule.setup()

        >>> # Access the training images
        >>> train_dataloader = folder_datamodule.train_dataloader()
        >>> batch = next(iter(train_dataloader))
        >>> print(batch.keys(), batch["image"].shape)

        >>> # Access the test images
        >>> test_dataloader = folder_datamodule.test_dataloader()
        >>> batch = next(iter(test_dataloader))
        >>> print(batch.keys(), batch["image"].shape)

    Note:
        The dataset is expected to have a structure similar to:

        .. code-block:: bash

            sample_dataset/
            ├── good/
            │   ├── normal_image1.jpg
            │   ├── normal_image2.jpg
            │   └── ...
            ├── crack/
            │   ├── anomaly_image1.jpg
            │   ├── anomaly_image2.jpg
            │   └── ...
            └── mask/
                └── crack/
                    ├── anomaly_mask1.png
                    ├── anomaly_mask2.png
                    └── ...
    """

    def __init__(
        self,
        name: str,
        normal_dir: str | Path | Sequence[str | Path],
        root: str | Path | None = None,
        abnormal_dir: str | Path | Sequence[str | Path] | None = None,
        normal_test_dir: str | Path | Sequence[str | Path] | None = None,
        mask_dir: str | Path | Sequence[str | Path] | None = None,
        normal_split_ratio: float = 0.2,
        extensions: tuple[str] | None = None,
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
        val_split_mode: ValSplitMode | str = ValSplitMode.FROM_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self._name = name
        self.root = root
        self.normal_dir = normal_dir
        self.abnormal_dir = abnormal_dir
        self.normal_test_dir = normal_test_dir
        self.mask_dir = mask_dir
        self.task = TaskType(task)
        self.extensions = extensions
        test_split_mode = TestSplitMode(test_split_mode)
        val_split_mode = ValSplitMode(val_split_mode)
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            image_size=image_size,
            transform=transform,
            train_transform=train_transform,
            eval_transform=eval_transform,
            seed=seed,
        )

        if task == TaskType.SEGMENTATION and test_split_mode == TestSplitMode.FROM_DIR and mask_dir is None:
            msg = (
                f"Segmentation task requires mask directory if test_split_mode is {test_split_mode}. "
                "You could set test_split_mode to {TestSplitMode.NONE} or provide a mask directory."
            )
            raise ValueError(
                msg,
            )

        self.normal_split_ratio = normal_split_ratio

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = FolderDataset(
            name=self.name,
            task=self.task,
            transform=self.train_transform,
            split=Split.TRAIN,
            root=self.root,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            normal_test_dir=self.normal_test_dir,
            mask_dir=self.mask_dir,
            extensions=self.extensions,
        )

        self.test_data = FolderDataset(
            name=self.name,
            task=self.task,
            transform=self.eval_transform,
            split=Split.TEST,
            root=self.root,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            normal_test_dir=self.normal_test_dir,
            mask_dir=self.mask_dir,
            extensions=self.extensions,
        )

    @property
    def name(self) -> str:
        """Name of the datamodule.

        Folder datamodule overrides the name property to provide a custom name.
        """
        return self._name
