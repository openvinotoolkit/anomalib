"""Anomalib datamodule base class."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightning.pytorch import LightningDataModule
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data.dataloader import DataLoader, default_collate
from torchvision.transforms.v2 import Resize, Transform

from anomalib.data.utils import TestSplitMode, ValSplitMode, random_split, split_by_label
from anomalib.data.utils.synthetic import SyntheticAnomalyDataset

if TYPE_CHECKING:
    from pandas import DataFrame

    from anomalib.data.base.dataset import AnomalibDataset

logger = logging.getLogger(__name__)


def collate_fn(batch: list) -> dict[str, Any]:
    """Collate bounding boxes as lists.

    Bounding boxes are collated as a list of tensors, while the default collate function is used for all other entries.

    Args:
        batch (List): list of items in the batch where len(batch) is equal to the batch size.

    Returns:
        dict[str, Any]: Dictionary containing the collated batch information.
    """
    elem = batch[0]  # sample an element from the batch to check the type.
    out_dict = {}
    if isinstance(elem, dict):
        if "boxes" in elem:
            # collate boxes as list
            out_dict["boxes"] = [item.pop("boxes") for item in batch]
        # collate other data normally
        out_dict.update({key: default_collate([item[key] for item in batch]) for key in elem})
        return out_dict
    return default_collate(batch)


class AnomalibDataModule(LightningDataModule, ABC):
    """Base Anomalib data module.

    Args:
        train_batch_size (int): Batch size used by the train dataloader.
        eval_batch_size (int): Batch size used by the val and test dataloaders.
        num_workers (int): Number of workers used by the train, val and test dataloaders.
        val_split_mode (ValSplitMode): Determines how the validation split is obtained.
            Options: [none, same_as_test, from_test, synthetic]
        val_split_ratio (float): Fraction of the train or test images held our for validation.
        test_split_mode (Optional[TestSplitMode], optional): Determines how the test split is obtained.
            Options: [none, from_dir, synthetic].
            Defaults to ``None``.
        test_split_ratio (float): Fraction of the train images held out for testing.
            Defaults to ``None``.
        image_size (tuple[int, int], optional): Size to which input images should be resized.
            Defaults to ``None``.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        train_transform (Transform, optional): Transforms that should be applied to the input images during training.
            Defaults to ``None``.
        eval_transform (Transform, optional): Transforms that should be applied to the input images during evaluation.
            Defaults to ``None``.
        seed (int | None, optional): Seed used during random subset splitting.
            Defaults to ``None``.
    """

    def __init__(
        self,
        train_batch_size: int,
        eval_batch_size: int,
        num_workers: int,
        val_split_mode: ValSplitMode | str,
        val_split_ratio: float,
        test_split_mode: TestSplitMode | str | None = None,
        test_split_ratio: float | None = None,
        image_size: tuple[int, int] | None = None,
        transform: Transform | None = None,
        train_transform: Transform | None = None,
        eval_transform: Transform | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.test_split_mode = TestSplitMode(test_split_mode) if test_split_mode else TestSplitMode.NONE
        self.test_split_ratio = test_split_ratio
        self.val_split_mode = ValSplitMode(val_split_mode)
        self.val_split_ratio = val_split_ratio
        self.image_size = image_size
        self.seed = seed

        # set transforms
        if bool(train_transform) != bool(eval_transform):
            msg = "Only one of train_transform and eval_transform was specified. This is not recommended because \
                    it could lead to unexpected behaviour. Please ensure training and eval transforms have the same \
                    reshape and normalization characteristics."
            logger.warning(msg)
        self._train_transform = train_transform or transform
        self._eval_transform = eval_transform or transform

        self.train_data: AnomalibDataset
        self.val_data: AnomalibDataset
        self.test_data: AnomalibDataset

        self._samples: DataFrame | None = None
        self._category: str = ""

        self._is_setup = False  # flag to track if setup has been called from the trainer

    @property
    def name(self) -> str:
        """Name of the datamodule."""
        return self.__class__.__name__

    def setup(self, stage: str | None = None) -> None:
        """Set up train, validation and test data.

        Args:
            stage: str | None:  Train/Val/Test stages.
                Defaults to ``None``.
        """
        has_subset = any(hasattr(self, subset) for subset in ["train_data", "val_data", "test_data"])
        if not has_subset or not self._is_setup:
            self._setup(stage)
            self._create_test_split()
            self._create_val_split()
            if isinstance(stage, TrainerFn):
                # only set the flag if the stage is a TrainerFn, which means the setup has been called from a trainer
                self._is_setup = True

    @abstractmethod
    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.

        Note:
            The stage argument is not used here. This is because, for a given instance of an AnomalibDataModule
            subclass, all three subsets are created at the first call of setup(). This is to accommodate the subset
            splitting behaviour of anomaly tasks, where the validation set is usually extracted from the test set, and
            the test set must therefore be created as early as the `fit` stage.

        """
        raise NotImplementedError

    @property
    def category(self) -> str:
        """Get the category of the datamodule."""
        return self._category

    @category.setter
    def category(self, category: str) -> None:
        """Set the category of the datamodule."""
        self._category = category

    def _create_test_split(self) -> None:
        """Obtain the test set based on the settings in the config."""
        if self.test_data.has_normal:
            # split the test data into normal and anomalous so these can be processed separately
            normal_test_data, self.test_data = split_by_label(self.test_data)
        elif self.test_split_mode != TestSplitMode.NONE:
            # when the user did not provide any normal images for testing, we sample some from the training set,
            # except when the user explicitly requested no test splitting.
            logger.info(
                "No normal test images found. Sampling from training set using a split ratio of %0.2f",
                self.test_split_ratio,
            )
            if self.test_split_ratio is not None:
                self.train_data, normal_test_data = random_split(self.train_data, self.test_split_ratio, seed=self.seed)

        if self.test_split_mode == TestSplitMode.FROM_DIR:
            self.test_data += normal_test_data
        elif self.test_split_mode == TestSplitMode.SYNTHETIC:
            self.test_data = SyntheticAnomalyDataset.from_dataset(normal_test_data)
        elif self.test_split_mode != TestSplitMode.NONE:
            msg = f"Unsupported Test Split Mode: {self.test_split_mode}"
            raise ValueError(msg)

    def _create_val_split(self) -> None:
        """Obtain the validation set based on the settings in the config."""
        if self.val_split_mode == ValSplitMode.FROM_TRAIN:
            # randomly sampled from train set
            self.train_data, self.val_data = random_split(
                self.train_data,
                self.val_split_ratio,
                label_aware=True,
                seed=self.seed,
            )
        elif self.val_split_mode == ValSplitMode.FROM_TEST:
            # randomly sampled from test set
            self.test_data, self.val_data = random_split(
                self.test_data,
                self.val_split_ratio,
                label_aware=True,
                seed=self.seed,
            )
        elif self.val_split_mode == ValSplitMode.SAME_AS_TEST:
            # equal to test set
            self.val_data = self.test_data
        elif self.val_split_mode == ValSplitMode.SYNTHETIC:
            # converted from random training sample
            self.train_data, normal_val_data = random_split(self.train_data, self.val_split_ratio, seed=self.seed)
            self.val_data = SyntheticAnomalyDataset.from_dataset(normal_val_data)
        elif self.val_split_mode != ValSplitMode.NONE:
            msg = f"Unknown validation split mode: {self.val_split_mode}"
            raise ValueError(msg)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get train dataloader."""
        return DataLoader(
            dataset=self.train_data,
            shuffle=True,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get validation dataloader."""
        return DataLoader(
            dataset=self.val_data,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get test dataloader."""
        return DataLoader(
            dataset=self.test_data,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """Use the test dataloader for inference unless overridden."""
        return self.test_dataloader()

    @property
    def transform(self) -> Transform:
        """Property that returns the user-specified transform for the datamodule, if any.

        This property is accessed by the engine to set the transform for the model. The eval_transform takes precedence
        over the train_transform, because the transform that we store in the model is the one that should be used during
        inference.
        """
        if self._eval_transform:
            return self._eval_transform
        return None

    @property
    def train_transform(self) -> Transform:
        """Get the transforms that will be passed to the train dataset.

        If the train_transform is not set, the engine will request the transform from the model.
        """
        if self._train_transform:
            return self._train_transform
        if getattr(self, "trainer", None) and self.trainer.lightning_module and self.trainer.lightning_module.transform:
            return self.trainer.lightning_module.transform
        if self.image_size:
            return Resize(self.image_size, antialias=True)
        return None

    @property
    def eval_transform(self) -> Transform:
        """Get the transform that will be passed to the val/test/predict datasets.

        If the eval_transform is not set, the engine will request the transform from the model.
        """
        if self._eval_transform:
            return self._eval_transform
        if getattr(self, "trainer", None) and self.trainer.lightning_module and self.trainer.lightning_module.transform:
            return self.trainer.lightning_module.transform
        if self.image_size:
            return Resize(self.image_size, antialias=True)
        return None

    @classmethod
    def from_config(
        cls: type["AnomalibDataModule"],
        config_path: str | Path,
        **kwargs,
    ) -> "AnomalibDataModule":
        """Create a datamodule instance from the configuration.

        Args:
            config_path (str | Path): Path to the data configuration file.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            AnomalibDataModule: Datamodule instance.

        Example:
            The following example shows how to get datamodule from mvtec.yaml:

            .. code-block:: python
                >>> data_config = "configs/data/mvtec.yaml"
                >>> datamodule = AnomalibDataModule.from_config(config_path=data_config)

            The following example shows overriding the configuration file with additional keyword arguments:

            .. code-block:: python
                >>> override_kwargs = {"data.train_batch_size": 8}
                >>> datamodule = AnomalibDataModule.from_config(config_path=data_config, **override_kwargs)
        """
        from jsonargparse import ArgumentParser

        if not Path(config_path).exists():
            msg = f"Configuration file not found: {config_path}"
            raise FileNotFoundError(msg)

        data_parser = ArgumentParser()
        data_parser.add_subclass_arguments(AnomalibDataModule, "data", required=False, fail_untyped=False)
        args = ["--data", str(config_path)]
        for key, value in kwargs.items():
            args.extend([f"--{key}", str(value)])
        config = data_parser.parse_args(args=args)
        instantiated_classes = data_parser.instantiate_classes(config)
        datamodule = instantiated_classes.get("data")
        if isinstance(datamodule, AnomalibDataModule):
            return datamodule

        msg = f"Datamodule is not an instance of AnomalibDataModule: {datamodule}"
        raise ValueError(msg)
