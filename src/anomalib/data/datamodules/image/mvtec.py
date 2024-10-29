"""MVTec AD Data Module.

Description:
    This script contains PyTorch Lightning DataModule for the MVTec AD dataset.
    If the dataset is not on the file system, the script downloads and extracts
    the dataset and create PyTorch data objects.

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
from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.mvtec import MVTecDataset
from anomalib.data.utils import DownloadInfo, Split, TestSplitMode, ValSplitMode, download_and_extract

logger = logging.getLogger(__name__)


DOWNLOAD_INFO = DownloadInfo(
    name="mvtec",
    url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094"
    "/mvtec_anomaly_detection.tar.xz",
    hashsum="cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d",
)


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
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
        )
        self.test_data = MVTecDataset(
            task=self.task,
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
