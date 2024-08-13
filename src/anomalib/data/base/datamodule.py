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

from anomalib.data.utils import SplitMode, TestSplitMode, ValSplitMode, resolve_split_mode
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
        val_split_mode: SplitMode | ValSplitMode | str | None = None,
        val_split_ratio: float | None = None,
        test_split_mode: SplitMode | TestSplitMode | str | None = None,
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
        self.test_split_ratio = test_split_ratio
        self.val_split_ratio = val_split_ratio
        self.image_size = image_size
        self.seed = seed

        # Check the split mode for backward compatibility
        self.test_split_mode = resolve_split_mode(test_split_mode)
        self.val_split_mode = resolve_split_mode(val_split_mode)

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

    @property
    def category(self) -> str:
        """Get the category of the datamodule."""
        return self._category

    @category.setter
    def category(self, category: str) -> None:
        """Set the category of the datamodule."""
        self._category = category

    def setup(self, stage: str | None = None) -> None:
        """Set up train, validation and test data.

        Args:
            stage: str | None:  Train/Val/Test stages.
                Defaults to ``None``.
        """
        # Check if setup is needed.
        has_any_dataset = any(hasattr(self, dataset) for dataset in ["train_data", "val_data", "test_data"])
        if has_any_dataset and self._is_setup:
            # Setup already completed
            # Validate the dataset splits meet the required criteria and exit early.
            self._validate_datasets()
            return

        # Perform implementation-specific setup
        self._setup(stage)

        # Post setup processing
        self._post_setup()

        # Set the flag if the stage is a TrainerFn, which means the setup has been called from a trainer
        if isinstance(stage, TrainerFn):
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

    def _post_setup(self) -> None:
        """Post setup method to process the datasets and validate the splits."""
        self._process_datasets()
        self._validate_datasets()

    def _process_datasets(self) -> None:
        """Process datasets based on the available datasets."""
        available_datasets = [split for split in ["train", "val", "test"] if hasattr(self, f"{split}_data")]
        logger.info(f"Available datasets: {available_datasets}")

        # We don't need to process the datasets if all three are available
        if available_datasets == ["train", "val", "test"]:
            return

        # Otherwise, process the datasets based on the available datasets
        if available_datasets == ["train"]:
            self._process_train_only_scenario()
        elif available_datasets == ["train", "test"]:
            self._process_train_test_scenario()
        elif available_datasets == ["train", "val"]:
            self._process_train_val_scenario()
        else:
            msg = "Invalid dataset configuration."
            raise ValueError(msg)

    def _process_train_only_scenario(self) -> None:
        if self.test_split_mode == SplitMode.AUTO:
            if self.train_data.has_anomalous:
                # 1. assign abnormal images to the eval set
                normal_dataset, abnormal_dataset = self.train_data.create_subset("label", seed=self.seed)

                # 2. split the normal dataset to train/test splits with test_split_ratio
                # if self.test_split_ratio is None, the default value is 0.4 (60% train, 40% test (20, 20 val/test))
                split_ratio = self.test_split_ratio or 0.4
                logger.info(f"Splitting normal images with ratio: {split_ratio}")
                self.train_data, normal_eval_dataset = normal_dataset.create_subset(
                    criteria=[1 - split_ratio, split_ratio],
                    seed=self.seed,
                )

                # 3. split the eval dataset to val/test splits with val_split_ratio
                eval_dataset = normal_eval_dataset + abnormal_dataset
                split_ratio = self.val_split_ratio or 0.5
                self.val_data, self.test_data = eval_dataset.create_subset(
                    criteria=[split_ratio, 1 - split_ratio],
                    label_aware=True,
                    seed=self.seed,
                )
            else:
                logger.warning(
                    "No abnormal images found in the train set. "
                    "Skipping val/test set creation. "
                    "This means that the model will not be evaluated.",
                )
        elif self.test_split_mode == SplitMode.PREDEFINED:
            logger.warning(
                "Skipping val/test set creation. This means that the model will not be evaluated.",
            )
            # if there are abnormal images in the train set warn the user.
            if self.train_data.has_anomalous:
                logger.warning(
                    "Train set contains abnormal images, but no val/test set is created. "
                    "If this is intended, you can ignore this warning.",
                )
        elif self.test_split_mode == SplitMode.SYNTHETIC:
            logger.info("Generating synthetic val and test sets.")
            self.val_data = SyntheticAnomalyDataset.from_dataset(self.train_data)
            self.test_data = SyntheticAnomalyDataset.from_dataset(self.train_data)

        else:
            msg = f"Invalid test split mode: {self.test_split_mode}"
            raise ValueError(msg)

    def _process_train_test_scenario(self) -> None:
        if self.val_split_mode == SplitMode.AUTO:
            if self.val_split_ratio is None:
                logger.info("'val_split_ratio' is not specified. Choosing a default value of 0.5 for AUTO mode.")
                split_ratio = 0.5
            else:
                split_ratio = self.val_split_ratio

            self.val_data, self.test_data = self.test_data.create_subset(
                criteria=[split_ratio, 1 - split_ratio],
                seed=self.seed,
                label_aware=True,
            )
        elif self.val_split_mode == SplitMode.PREDEFINED:
            logger.warning(
                "Skipping val set creation. This means that the model will not be evaluated."
                "You can use 'SplitMode.AUTO' to automatically create a val set by randomly sampling from the test set"
                "or 'SplitMode.SYNTHETIC' to generate synthetic val set.",
            )
        elif self.val_split_mode == SplitMode.SYNTHETIC:
            logger.info("Generating synthetic val set.")
            self.val_data = SyntheticAnomalyDataset.from_dataset(self.train_data)
        else:
            msg = f"Invalid val split mode: {self.val_split_mode}"
            raise ValueError(msg)

    def _process_train_val_scenario(self) -> None:
        if self.test_split_mode == SplitMode.AUTO:
            if self.test_split_ratio is None:
                logger.info("'test_split_ratio' is not specified. Choosing a default value of 0.5 for AUTO mode.")
                split_ratio = 0.5
            else:
                split_ratio = self.test_split_ratio

            self.val_data, self.test_data = self.val_data.create_subset(
                criteria=[1 - split_ratio, split_ratio],
                seed=self.seed,
                label_aware=True,
            )
        elif self.test_split_mode == SplitMode.PREDEFINED:
            logger.warning(
                "Skipping test set creation. This means that the model will not be evaluated."
                "You can use 'SplitMode.AUTO' to automatically create a test set, "
                "or 'SplitMode.SYNTHETIC' to generate synthetic test set.",
            )
        elif self.test_split_mode == SplitMode.SYNTHETIC:
            logger.info("Generating synthetic test set.")
            self.test_data = SyntheticAnomalyDataset.from_dataset(self.train_data)
        else:
            msg = f"Invalid test split mode: {self.test_split_mode}"
            raise ValueError(msg)

    def _validate_datasets(self) -> None:
        """Perform sanity check on train, validation, and test sets."""
        # Check train set
        if hasattr(self, "train_data") and not self.train_data.all_normal:
            msg = "Train set should contain only normal images."
            raise ValueError(msg)

        # Check validation set
        if hasattr(self, "val_data") and not (self.val_data.has_normal and self.val_data.has_anomalous):
            msg = "Validation set should contain both normal and abnormal images."
            raise ValueError(msg)

        # Check test set
        if hasattr(self, "test_data") and not (self.test_data.has_normal and self.test_data.has_anomalous):
            msg = "Test set should contain both normal and abnormal images."
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
        if getattr(self, "trainer", None) and self.trainer.model and self.trainer.model.transform:
            return self.trainer.model.transform
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
        if getattr(self, "trainer", None) and self.trainer.model and self.trainer.model.transform:
            return self.trainer.model.transform
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
