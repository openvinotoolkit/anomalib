"""Anomalib datamodule base class."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from lightning.pytorch import LightningDataModule
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data.dataloader import DataLoader

from anomalib.data.utils import SplitMode, TestSplitMode, ValSplitMode, resolve_split_mode
from anomalib.data.utils.synthetic import SyntheticAnomalyDataset

if TYPE_CHECKING:
    from pandas import DataFrame

    from anomalib.data.datasets.base.image import AnomalibDataset

logger = logging.getLogger(__name__)


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
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.test_split_ratio = test_split_ratio
        self.val_split_ratio = val_split_ratio
        self.seed = seed

        # Check the split mode for backward compatibility
        self.test_split_mode = resolve_split_mode(test_split_mode)
        self.val_split_mode = resolve_split_mode(val_split_mode)

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
                logger.info(f"Splitting normal images to train/test splits with ratio: {split_ratio}")
                normal_eval_dataset, self.train_data = normal_dataset.create_subset(
                    criteria=split_ratio,
                    seed=self.seed,
                )

                # 3. split the eval dataset to val/test splits with val_split_ratio
                eval_dataset = normal_eval_dataset + abnormal_dataset
                split_ratio = self.val_split_ratio or 0.5
                logger.info(f"Splitting eval images to val/test splits with ratio: {split_ratio}")
                self.val_data, self.test_data = eval_dataset.create_subset(
                    criteria=split_ratio,
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
            # First split the train set to train/eval splits to get different
            # normal images for train and eval sets.
            split_ratio = self.test_split_ratio or 0.4
            logger.info(f"Splitting normal train images to train/eval splits with ratio: {split_ratio}")
            eval_train_data, self.train_data = self.train_data.create_subset(
                criteria=split_ratio,
                seed=self.seed,
            )

            # Generate synthetic val and test sets
            logger.info("Generating synthetic val and test sets.")
            synthetic_eval_data = SyntheticAnomalyDataset.from_dataset(eval_train_data)

            # Split the eval data to val/test splits with val_split_ratio
            split_ratio = self.val_split_ratio or 0.5
            logger.info(f"Splitting synthetic eval images to val/test splits with ratio: {split_ratio}")
            self.val_data, self.test_data = synthetic_eval_data.create_subset(
                criteria=split_ratio,
                seed=self.seed,
                label_aware=True,
            )

        else:
            msg = f"Invalid test split mode: {self.test_split_mode}"
            raise ValueError(msg)

    def _process_train_test_scenario(self) -> None:
        if self.val_split_ratio is None:
            logger.info("'val_split_ratio' is not specified. Choosing a default value of 0.5 for AUTO mode.")
            split_ratio = 0.5
        else:
            split_ratio = self.val_split_ratio

        if self.val_split_mode == SplitMode.AUTO:
            self.val_data, self.test_data = self.test_data.create_subset(
                criteria=split_ratio,
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
            normal_val_data, self.train_data = self.train_data.create_subset(
                criteria=split_ratio,
                seed=self.seed,
            )
            logger.info("Generating synthetic val set.")
            self.val_data = SyntheticAnomalyDataset.from_dataset(normal_val_data)
        else:
            msg = f"Invalid val split mode: {self.val_split_mode}"
            raise ValueError(msg)

    def _process_train_val_scenario(self) -> None:
        if self.test_split_ratio is None:
            logger.info("'test_split_ratio' is not specified. Choosing a default value of 0.5 for AUTO mode.")
            split_ratio = 0.5
        else:
            split_ratio = self.test_split_ratio

        if self.test_split_mode == SplitMode.AUTO:
            self.test_data, self.val_data = self.val_data.create_subset(
                criteria=split_ratio,
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
            normal_test_data, self.train_data = self.train_data.create_subset(
                criteria=split_ratio,
                seed=self.seed,
            )
            logger.info("Generating synthetic test set.")
            self.test_data = SyntheticAnomalyDataset.from_dataset(normal_test_data)
        else:
            msg = f"Invalid test split mode: {self.test_split_mode}"
            raise ValueError(msg)

    def _validate_datasets(self) -> None:
        """Perform sanity check on train, validation, and test sets."""
        # Check train set
        if hasattr(self, "train_data") and not self.train_data.all_normal:
            logger.warning(
                "Train set contains abnormal images. "
                "This is unusual and may lead to unexpected results. "
                "Typically, the train set should contain only normal images.",
            )

        # Check validation set
        if hasattr(self, "val_data") and not (self.val_data.has_normal and self.val_data.has_anomalous):
            logger.warning(
                "Validation set does not contain both normal and abnormal images. "
                "This may impact the ability to properly evaluate the model after training. "
                "It's recommended to have both normal and abnormal images in the validation set.",
            )

        # Check test set
        if hasattr(self, "test_data") and not (self.test_data.has_normal and self.test_data.has_anomalous):
            logger.warning(
                "Test set does not contain both normal and abnormal images. "
                "This may lead to incomplete or biased evaluation of the model. "
                "It's recommended to have both normal and abnormal images in the test set.",
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get train dataloader."""
        return DataLoader(
            dataset=self.train_data,
            shuffle=True,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.train_data.collate_fn,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get validation dataloader."""
        return DataLoader(
            dataset=self.val_data,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.val_data.collate_fn,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get test dataloader."""
        return DataLoader(
            dataset=self.test_data,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.test_data.collate_fn,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """Use the test dataloader for inference unless overridden."""
        return self.test_dataloader()

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
