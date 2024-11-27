"""DataModule for Datumaro format.

Note: This currently only works for annotations exported from Intel Geti™.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from anomalib import TaskType
from anomalib.data.datamodules.base import AnomalibDataModule
from anomalib.data.datasets.image.datumaro import DatumaroDataset
from anomalib.data.utils import Split, SplitMode, TestSplitMode, ValSplitMode


class Datumaro(AnomalibDataModule):
    """Datumaro datamodule.

    Args:
        root (str | Path): Path to the dataset root directory.
        train_batch_size (int): Batch size for training dataloader.
            Defaults to ``32``.
        eval_batch_size (int): Batch size for evaluation dataloader.
            Defaults to ``32``.
        num_workers (int): Number of workers for dataloaders.
            Defaults to ``8``.
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``.
            Defaults to ``TaskType.CLASSIFICATION``. Currently only supports classification.
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
        To create a Datumaro datamodule

        >>> from pathlib import Path
        >>> from torchvision.transforms.v2 import Resize
        >>> root = Path("path/to/dataset")
        >>> datamodule = Datumaro(root, transform=Resize((256, 256)))
        >>> datamodule.setup()
        >>> i, data = next(enumerate(datamodule.train_dataloader()))
        >>> data.keys()
        dict_keys(['image_path', 'label', 'image'])

        >>> data["image"].shape
        torch.Size([32, 3, 256, 256])
    """

    def __init__(
        self,
        root: str | Path,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType = TaskType.CLASSIFICATION,
        test_split_mode: SplitMode | TestSplitMode | str = SplitMode.AUTO,
        test_split_ratio: float | None = None,
        val_split_mode: SplitMode | ValSplitMode | str = SplitMode.AUTO,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        if task != TaskType.CLASSIFICATION:
            msg = "Datumaro dataloader currently only supports classification task."
            raise ValueError(msg)
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            seed=seed,
        )
        self.root = root
        self.task = task

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = DatumaroDataset(
            task=self.task,
            root=self.root,
            split=Split.TRAIN,
        )
        self.test_data = DatumaroDataset(
            task=self.task,
            root=self.root,
            split=Split.TEST,
        )
