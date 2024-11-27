"""Custom Folder Data Module.

This script creates a custom Lightning DataModule from a folder.
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from pathlib import Path

from anomalib import TaskType
from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.folder import FolderDataset
from anomalib.data.utils import SplitMode, TestSplitMode, ValSplitMode, resolve_split_mode


class Folder(AnomalibDataModule):
    """Folder DataModule for handling folder-based anomaly detection datasets.

    This DataModule is designed to work with datasets organized in folder structures, where
    normal and abnormal images are stored in separate directories. It supports various
    anomaly detection tasks and can handle mask annotations for segmentation tasks.

    The expected folder structure for a typical anomaly detection dataset is as follows:

    .. code-block:: text

        dataset_root/
        ├── normal/
        │   ├── normal_image_1.png
        │   ├── normal_image_2.png
        │   └── ...
        ├── abnormal/
        │   ├── abnormal_category_1/
        │   │   ├── abnormal_image_1.png
        │   │   ├── abnormal_image_2.png
        │   │   └── ...
        │   ├── abnormal_category_2/
        │   │   ├── abnormal_image_1.png
        │   │   ├── abnormal_image_2.png
        │   │   └── ...
        │   └── ...
        └── masks/  # Optional, for segmentation tasks
            ├── abnormal_category_1/
            │   ├── abnormal_image_1_mask.png
            │   ├── abnormal_image_2_mask.png
            │   └── ...
            ├── abnormal_category_2/
            │   ├── abnormal_image_1_mask.png
            │   ├── abnormal_image_2_mask.png
            │   └── ...
            └── ...

    Args:
        name (str): Name of the dataset.
        normal_dir (str | Path | Sequence[str | Path]): Path(s) to the directory(ies) containing normal images.
        abnormal_dir (str | Path | Sequence[str | Path] | None, optional): Path(s) to the directory(ies)
            containing abnormal images.
            Defaults to ``None``.
        mask_dir (str | Path | Sequence[str | Path] | None, optional): Path(s) to the directory(ies)
            containing mask annotations for segmentation tasks.
            Defaults to ``None``.
        root (str | Path | None, optional): Root directory of the dataset. If provided, other paths will be
            resolved relative to this. Defaults to ``None``.
        extensions (tuple[str] | None, optional): File extensions to include when reading images.
            If None, all files will be considered. Defaults to ``None``.
        train_batch_size (int, optional): Batch size for training.
            Defaults to ``32``.
        eval_batch_size (int, optional): Batch size for evaluation.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers for data loading.
            Defaults to ``8``.
        task (TaskType | str, optional): Type of the anomaly detection task.
            Defaults to ``TaskType.SEGMENTATION``.
        test_split_mode (SplitMode | str, optional): Mode for creating the test split.
            Defaults to ``None``.
        test_split_mode (SplitMode | str, optional): Mode for creating the test split.
            Defaults to ``SplitMode.AUTO``.
        test_split_ratio (float | None, optional): Ratio of data to use for testing.
            Defaults to ``None``.
        val_split_mode (SplitMode | str, optional): Mode for creating the validation split.
            Defaults to ``SplitMode.AUTO``.
        val_split_ratio (float, optional): Ratio of training data to use for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Random seed for reproducibility.
            Defaults to ``None``.

    Attributes:
        SplitMode (Enum): Enumeration of available split modes:
            - SYNTHETIC: Generate synthetic data for splitting.
            - PREDEFINED: Use a pre-defined split from an existing source.
            - AUTO: Automatically determine the best splitting strategy.

    Examples:
        Basic usage with normal and abnormal directories:

        >>> from anomalib.data import Folder
        >>> datamodule = Folder(
        ...     name="my_dataset",
        ...     normal_dir="path/to/normal",
        ...     abnormal_dir="path/to/abnormal",
        ...     task="classification"
        ... )

        Using multiple directories for normal and abnormal images:

        >>> datamodule = Folder(
        ...     name="multi_dir_dataset",
        ...     normal_dir=["path/to/normal1", "path/to/normal2"],
        ...     abnormal_dir=["path/to/abnormal1", "path/to/abnormal2"],
        ...     task="classification"
        ... )

        Segmentation task with mask directory:

        >>> datamodule = Folder(
        ...     name="segmentation_dataset",
        ...     normal_dir="path/to/normal",
        ...     abnormal_dir="path/to/abnormal",
        ...     mask_dir="path/to/masks",
        ...     task="segmentation"
        ... )

        Customizing data splits using the new SplitMode:

        >>> from anomalib.data.utils import SplitMode
        >>> datamodule = Folder(
        ...     name="custom_split_dataset",
        ...     normal_dir="path/to/normal",
        ...     abnormal_dir="path/to/abnormal",
        ...     test_split_mode=SplitMode.AUTO,
        ...     val_split_mode=SplitMode.SYNTHETIC,
        ...     test_split_ratio=0.2,
        ...     val_split_ratio=0.5
        ... )

    Note:
        - For segmentation tasks, ensure that the mask_dir is provided and
          contains corresponding masks for abnormal images.
        - The class supports both single directory and multiple directory inputs
          for normal, abnormal, and mask data.
        - The old TestSplitMode and ValSplitMode enums are deprecated. Use the new
          SplitMode enum instead.
        - When using SplitMode.PREDEFINED for segmentation tasks, a mask_dir must be provided.

    Warning:
        The 'normal_test_dir' and 'normal_split_ratio' arguments are deprecated and will be
        removed in a future release. Use 'test_split_ratio' or 'val_split_ratio' with the
        appropriate SplitMode instead.
    """

    def __init__(
        self,
        name: str,
        normal_dir: str | Path | Sequence[str | Path],
        abnormal_dir: str | Path | Sequence[str | Path] | None = None,
        mask_dir: str | Path | Sequence[str | Path] | None = None,
        root: str | Path | None = None,
        extensions: tuple[str] | None = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType | str = TaskType.SEGMENTATION,
        test_split_mode: SplitMode | TestSplitMode | str = SplitMode.AUTO,
        test_split_ratio: float | None = None,
        val_split_mode: SplitMode | ValSplitMode | str = SplitMode.AUTO,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
        normal_test_dir: str | Path | Sequence[str | Path] | None = None,  # DEPRECATED
        normal_split_ratio: float | None = None,  # DEPRECATED
    ) -> None:
        self._name = name
        self.root = root
        self.normal_dir = normal_dir
        self.abnormal_dir = abnormal_dir
        self.mask_dir = mask_dir
        self.task = TaskType(task)
        self.extensions = extensions
        test_split_mode = resolve_split_mode(test_split_mode)
        val_split_mode = resolve_split_mode(val_split_mode)
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

        if normal_test_dir is not None:
            msg = (
                "The 'normal_test_dir' argument is deprecated and will be removed in a future release. "
                "If you have a dedicated train/val/test directories, please use CSV datamodule instead."
            )
            raise DeprecationWarning(msg)

        if normal_split_ratio is not None:
            msg = (
                "The 'normal_split_ratio' argument is deprecated and will be removed in a future release. "
                "Please use 'test_split_ratio' or 'val_split_ratio' instead."
            )
            raise DeprecationWarning(msg)

        if task == TaskType.SEGMENTATION and test_split_mode == TestSplitMode.FROM_DIR and mask_dir is None:
            msg = (
                f"Segmentation task requires mask directory if test_split_mode is {test_split_mode}. "
                "You could set test_split_mode to {TestSplitMode.NONE} or provide a mask directory."
            )
            raise ValueError(msg)

    def _setup(self, _stage: str | None = None) -> None:
        """Setup the Folder datamodule.

        By default, the Folder datamodule auto splits the dataset into train/val/test.
        The split will be handled by `post_setup` method in the base class.
        """
        self.train_data = FolderDataset(
            name=self.name,
            task=self.task,
            root=self.root,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            mask_dir=self.mask_dir,
            extensions=self.extensions,
        )

    @property
    def name(self) -> str:
        """Name of the datamodule.

        Folder datamodule overrides the name property to provide a custom name.
        """
        return self._name
