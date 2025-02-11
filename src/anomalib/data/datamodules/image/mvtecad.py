"""MVTec AD Data Module.

This module provides a PyTorch Lightning DataModule for the MVTec AD dataset. If
the dataset is not available locally, it will be downloaded and extracted
automatically.

Example:
    Create a MVTec AD datamodule::

        >>> from anomalib.data import MVTecAD
        >>> datamodule = MVTecAD(
        ...     root="./datasets/MVTecAD",
        ...     category="bottle"
        ... )

Notes:
    The dataset will be automatically downloaded and converted to the required
    format when first used. The directory structure after preparation will be::

        datasets/
        └── MVTecAD/
            ├── bottle/
            ├── cable/
            └── ...

License:
    MVTec AD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0).
    https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger,
    Carsten Steger: The MVTec Anomaly Detection Dataset: A Comprehensive
    Real-World Dataset for Unsupervised Anomaly Detection; in: International
    Journal of Computer Vision 129(4):1038-1059, 2021,
    DOI: 10.1007/s11263-020-01400-4.

    Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: MVTec AD —
    A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection;
    in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
    9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.mvtecad import MVTecADDataset
from anomalib.data.utils import DownloadInfo, Split, TestSplitMode, ValSplitMode, download_and_extract

logger = logging.getLogger(__name__)


DOWNLOAD_INFO = DownloadInfo(
    name="mvtecad",
    url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/"
    "download/420938113-1629952094/mvtec_anomaly_detection.tar.xz",
    hashsum="cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d",
)


class MVTecAD(AnomalibDataModule):
    """MVTec AD Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/MVTecAD"``.
        category (str): Category of the MVTec AD dataset (e.g. ``"bottle"`` or
            ``"cable"``). Defaults to ``"bottle"``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply dto the training images
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
        test_split_mode (TestSplitMode): Method to create test set.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of data to use for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Method to create validation set.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of data to use for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed for reproducibility.
            Defaults to ``None``.

    Example:
        Create MVTec AD datamodule with default settings::

            >>> datamodule = MVTecAD()
            >>> datamodule.setup()
            >>> i, data = next(enumerate(datamodule.train_dataloader()))
            >>> data.keys()
            dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

            >>> data["image"].shape
            torch.Size([32, 3, 256, 256])

        Change the category::

            >>> datamodule = MVTecAD(category="cable")

        Create validation set from test data::

            >>> datamodule = MVTecAD(
            ...     val_split_mode=ValSplitMode.FROM_TEST,
            ...     val_split_ratio=0.1
            ... )

        Create synthetic validation set::

            >>> datamodule = MVTecAD(
            ...     val_split_mode=ValSplitMode.SYNTHETIC,
            ...     val_split_ratio=0.2
            ... )
    """

    def __init__(
        self,
        root: Path | str = "./datasets/MVTecAD",
        category: str = "bottle",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
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

        self.root = Path(root)
        self.category = category

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.

        Note:
            The stage argument is not used here. This is because, for a given
            instance of an AnomalibDataModule subclass, all three subsets are
            created at the first call of setup(). This is to accommodate the
            subset splitting behaviour of anomaly tasks, where the validation set
            is usually extracted from the test set, and the test set must
            therefore be created as early as the `fit` stage.
        """
        self.train_data = MVTecADDataset(
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
        )
        self.test_data = MVTecADDataset(
            split=Split.TEST,
            root=self.root,
            category=self.category,
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available.

        This method checks if the specified dataset is available in the file
        system. If not, it downloads and extracts the dataset into the
        appropriate directory.

        Example:
            Assume the dataset is not available on the file system::

                >>> datamodule = MVTecAD(
                ...     root="./datasets/MVTecAD",
                ...     category="bottle"
                ... )
                >>> datamodule.prepare_data()

            Directory structure after download::

                datasets/
                └── MVTecAD/
                    ├── bottle/
                    ├── cable/
                    └── ...
        """
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)


class MVTec(MVTecAD):
    """MVTec datamodule class (Deprecated).

    This class is deprecated and will be removed in a future version.
    Please use MVTecAD instead.
    """

    def __init__(self, *args, **kwargs) -> None:
        import warnings

        warnings.warn(
            "MVTec is deprecated and will be removed in a future version. Please use MVTecAD instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
