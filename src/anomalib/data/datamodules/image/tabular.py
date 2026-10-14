# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom Tabular Data Module.

This script creates a custom Lightning DataModule from a table or tabular file
containing image paths and labels.

Example:
    Create a Tabular datamodule::

        >>> from anomalib.data import Tabular
        >>> samples = {
        ...     "image_path": ["images/image1.png", "images/image2.png", "images/image3.png", ... ],
        ...     "label_index": [LabelName.NORMAL, LabelName.NORMAL, LabelName.ABNORMAL,  ... ],
        ...     "split": [Split.TRAIN, Split.TRAIN, Split.TEST, ... ],
        ... }
        >>> datamodule = Tabular(
        ...     name="custom",
        ...     samples=samples,
        ...     root="./datasets/custom",
        ... )
"""

from pathlib import Path

import pandas as pd
from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.tabular import TabularDataset
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode

# Allowlist of file formats supported by ``Tabular.from_file``.
#
# This is intentionally a minimal, explicit allowlist rather than a
# passthrough to any ``pd.read_*`` method. Formats such as ``pickle`` and
# ``hdf`` are excluded because loading them can execute arbitrary code
# embedded in the file during deserialization. Additional safe, data-only
# formats can be added in the future as needed.
SUPPORTED_FILE_FORMATS = ("csv", "json", "parquet")


class Tabular(AnomalibDataModule):
    """Tabular DataModule.

    Args:
        name (str): Name of the dataset. Used for logging/saving.
        samples (dict | list | DataFrame): Pandas ``DataFrame`` or compatible ``list``
            or ``dict`` containing the dataset information.
        root (str | Path | None): Root folder containing normal and abnormal
            directories. Defaults to ``None``.
        normal_split_ratio (float): Ratio to split normal training images for
            test set when no normal test images exist.
            Defaults to ``0.2``.
        train_batch_size (int): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int): Validation/test batch size.
            Defaults to ``32``.
        num_workers (int): Number of workers for data loading.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply to the training images
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
        test_split_mode (TestSplitMode): Method to obtain test subset.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of train images for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Method to obtain validation subset.
            Defaults to ``ValSplitMode.FROM_TEST``.
        val_split_ratio (float): Fraction of images for validation.
            Defaults to ``0.5``.
        seed (int | None): Random seed for splitting.
            Defaults to ``None``.

    Example:
        Create and setup a tabular datamodule::

            >>> from anomalib.data import Tabular
            >>> samples = {
            ...     "image_path": ["images/image1.png", "images/image2.png", "images/image3.png", ... ],
            ...     "label_index": [LabelName.NORMAL, LabelName.NORMAL, LabelName.ABNORMAL,  ... ],
            ...     "split": [Split.TRAIN, Split.TRAIN, Split.TEST, ... ],
            ... }
            >>> datamodule = Tabular(
            ...     name="custom",
            ...     samples=samples,
            ...     root="./datasets/custom",
            ... )
            >>> datamodule.setup()

        Get a batch from train dataloader::

            >>> batch = next(iter(datamodule.train_dataloader()))
            >>> batch.keys()
            dict_keys(['image', 'label', 'mask', 'image_path', 'mask_path'])

        Get a batch from test dataloader::

            >>> batch = next(iter(datamodule.test_dataloader()))
            >>> batch.keys()
            dict_keys(['image', 'label', 'mask', 'image_path', 'mask_path'])
    """

    def __init__(
        self,
        name: str,
        samples: dict | list | pd.DataFrame,
        root: str | Path | None = None,
        normal_split_ratio: float = 0.2,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.FROM_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self._name = name
        self.root = root
        self._unprocessed_samples = samples
        test_split_mode = TestSplitMode(test_split_mode)
        val_split_mode = ValSplitMode(val_split_mode)
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.normal_split_ratio = normal_split_ratio

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = TabularDataset(
            name=self.name,
            samples=self._unprocessed_samples,
            split=Split.TRAIN,
            root=self.root,
        )

        self.test_data = TabularDataset(
            name=self.name,
            samples=self._unprocessed_samples,
            split=Split.TEST,
            root=self.root,
        )

    @property
    def name(self) -> str:
        """Get name of the datamodule.

        Returns:
            Name of the datamodule.
        """
        return self._name

    @classmethod
    def from_file(
        cls: type["Tabular"],
        name: str,
        file_path: str | Path,
        file_format: str | None = None,
        pd_kwargs: dict | None = None,
        **kwargs,
    ) -> "Tabular":
        """Create Tabular Datamodule from file.

        Args:
            name (str): Name of the dataset. This is used to name the datamodule,
                especially when logging/saving.
            file_path (str | Path): Path to tabular file containing the datset
                information.
            file_format (str | None): File format of the tabular file. Must be one of
                ``csv``, ``json`` or ``parquet`` (case-insensitive).
                Defaults to ``None`` (inferred from the file suffix).
            pd_kwargs (dict | None): Keyword argument dictionary for the pd.read_* method.
                Defaults to ``None``.
            kwargs (dict): Additional keyword arguments for the Tabular Datamodule class.

        Returns:
            Tabular: Tabular Datamodule

        Raises:
            FileNotFoundError: If ``file_path`` does not point to an existing file.
            ValueError: If ``file_format`` is not specified and cannot be inferred from
                the file name, or if it is not one of the supported formats.

        Note:
            Only the ``csv``, ``json`` and ``parquet`` formats are supported. Other
            ``pandas`` readers, such as ``pickle`` or ``hdf``, are intentionally not
            supported because loading them can execute arbitrary code embedded in the
            file during deserialization. To load data from such a format, read it into
            a ``DataFrame`` yourself and pass it to the ``Tabular`` constructor instead.

        Example:
            Prepare a tabular file (such as ``samples.csv`` or ``samples.parquet``) with the
            following columns: ``image_path`` (absolute or relative to ``root``), ``label_index``
            (``0`` for normal, ``1`` for anomalous samples), and ``split`` (``train`` or ``test``).
            For segmentation tasks, also include a ``mask_path`` column.

            From this file, create and setup a tabular datamodule::

                >>> from anomalib.data import Tabular
                >>> datamodule = Tabular.from_file(
                ...     name="custom",
                ...     file_path="./samples.csv",
                ...     root="./datasets/custom",
                ... )
                >>> datamodule.setup()

            Get a batch from train dataloader::

                >>> batch = next(iter(datamodule.train_dataloader()))
                >>> batch.keys()
                dict_keys(['image', 'label', 'mask', 'image_path', 'mask_path'])

            Get a batch from test dataloader::

                >>> batch = next(iter(datamodule.test_dataloader()))
                >>> batch.keys()
                dict_keys(['image', 'label', 'mask', 'image_path', 'mask_path'])
        """
        # Check if file exists
        if not Path(file_path).is_file():
            msg = f"File not found: '{file_path}'"
            raise FileNotFoundError(msg)

        # Infer file_format and check if supported
        file_format = (file_format or Path(file_path).suffix[1:]).lower()
        if not file_format:
            msg = f"File format not specified and could not be inferred from file name: '{Path(file_path).name}'"
            raise ValueError(msg)
        if file_format not in SUPPORTED_FILE_FORMATS:
            msg = (
                f"Unsupported file format: '{file_format}'. "
                f"Supported formats are: {', '.join(SUPPORTED_FILE_FORMATS)}. "
                "Formats such as 'pickle' and 'hdf' are not supported because loading "
                "them can execute arbitrary code. To use such a file, load it yourself "
                "and pass the resulting DataFrame to the Tabular constructor instead."
            )
            raise ValueError(msg)
        read_func = getattr(pd, f"read_{file_format}")

        # Read the file and return Tabular dataset
        pd_kwargs = pd_kwargs or {}
        samples = read_func(file_path, **pd_kwargs)
        return cls(name, samples, **kwargs)
