"""Anomalib datamodule base class."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightning.pytorch import LightningDataModule
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.v2 import Compose, Resize, Transform

from anomalib import TaskType
from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.utils import TestSplitMode, ValSplitMode, random_split, split_by_label
from anomalib.data.utils.synthetic import SyntheticAnomalyDataset

if TYPE_CHECKING:
    from pandas import DataFrame


logger = logging.getLogger(__name__)


class AnomalibDataModule(LightningDataModule, ABC):
    """Base Anomalib data module.

    Args:
        train_batch_size (int): Batch size used by the train dataloader.
        eval_batch_size (int): Batch size used by the val and test dataloaders.
        num_workers (int): Number of workers used by the train, val and test dataloaders.
        train_augmentations (Transform | None): Augmentations to apply dto the training images
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
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
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        val_split_mode: ValSplitMode | str | None = None,
        val_split_ratio: float | None = None,
        test_split_mode: TestSplitMode | str | None = None,
        test_split_ratio: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.test_split_mode = TestSplitMode(test_split_mode) if test_split_mode else TestSplitMode.NONE
        self.test_split_ratio = test_split_ratio or 0.5
        self.val_split_mode = ValSplitMode(val_split_mode)
        self.val_split_ratio = val_split_ratio or 0.5
        self.seed = seed

        self.train_augmentations = train_augmentations or augmentations
        self.val_augmentations = val_augmentations or augmentations
        self.test_augmentations = test_augmentations or augmentations

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

        self._update_augmentations()

    def _update_augmentations(self) -> None:
        """Update the augmentations for each subset."""
        for subset_name in ["train", "val", "test"]:
            subset = getattr(self, f"{subset_name}_data", None)
            augmentations = getattr(self, f"{subset_name}_augmentations", None)
            model_transform = self.get_nested_attr(self, "trainer.model.pre_processor.transform")
            if subset and augmentations:
                self._update_subset_augmentations(subset, augmentations, model_transform)

    def _update_subset_augmentations(
        self,
        dataset: AnomalibDataset,
        augmentations: Transform,
        model_transform: Transform,
    ) -> None:
        """Update the augmentations of the dataset.

        This method passes the user-specified augmentations to a dataset subset. If the model transforms contain
        a Resize transform, it will be appended to the augmentations. This will ensure that resizing takes place
        before collating, which reduces the usage of shared memory by the Dataloader workers.

        Args:
            dataset (AnomalibDataset): Dataset to update.
            augmentations (Transform): Augmentations to apply to the dataset.
            model_transform (Transform): Transform object from the model PreProcessor.
        """
        model_resizes = self.get_resize_transforms(model_transform)

        if model_resizes:
            model_resize = model_resizes[0]
            for aug_resize in self.get_resize_transforms(augmentations):  # warn user if resizes inconsistent
                if model_resize.size != aug_resize.size:
                    msg = f"Conflicting resize shapes found between augmentations and model transforms. You are using \
                        a Resize transform in your input data augmentations. Please be aware that the model also \
                        applies a Resize transform with a different output size. The final effective input size as \
                        seen by the model will be determined by the model transforms, not the augmentations. To change \
                        the effective input size, please change the model transforms in the PreProcessor module. \
                        Augmentations: {aug_resize.size}, Model transforms: {model_transform.size}"
                    logger.warning(msg)
                if model_resize.interpolation != aug_resize.interpolation:
                    msg = f"Conflicting interpolation method found between augmentations and model transforms. You are \
                        using a Resize transform in your input data augmentations. Please be aware that the model also \
                        applies a Resize transform with a different interpolation method. Using multiple interpolation \
                        methods can lead to unexpected behaviour, so it is recommended to use the same interpolation \
                        method between augmentations and model transforms. Augmentations: {aug_resize.interpolation}, \
                        Model transforms: {model_resize.interpolation}"
                    logger.warning(msg)
                if model_resize.antialias != aug_resize.antialias:
                    msg = f"Conflicting antialiasing setting found between augmentations and model transforms. You are \
                        using a Resize transform in your input data augmentations. Please be aware that the model also \
                        applies a Resize transform with a different antialising setting. Using conflicting \
                        antialiasing settings can lead to unexpected behaviour, so it is recommended to use the same \
                        antialiasing setting between augmentations and model transforms. Augmentations: \
                        antialias={aug_resize.antialias}, Model transforms: antialias={model_resize.antialias}"

            # append model resize to augmentations
            if isinstance(augmentations, Compose):
                augmentations = Compose([*augmentations.transforms, model_resize])
            elif isinstance(augmentations, Transform):
                augmentations = Compose([augmentations, model_resize])
            elif augmentations is None:
                augmentations = model_resize

        dataset.augmentations = augmentations

    @staticmethod
    def get_resize_transforms(transform: Transform | None) -> list[Resize]:
        """Get a list of all the resize transforms present in the provided Transform.

        Args:
            transform (Transform): Torchvision Transform instance.

        Returns:
            List[Resize]: List of Resize transform instances.
        """
        if isinstance(transform, Resize):
            return [transform]
        if isinstance(transform, Compose):
            return [transform for transform in transform.transforms if isinstance(transform, Resize)]
        return []

    @staticmethod
    def get_nested_attr(obj: Any, attr_path: str, default: Any | None = None) -> Any:  # noqa: ANN401
        """Safely retrieves a nested attribute from an object.

        Args:
            obj: The object to retrieve the attribute from.
            attr_path: A dot-separated string representing the attribute path.
            default: The default value to return if any attribute in the path is missing.

        Returns:
            The value of the nested attribute, or `default` if any attribute in the path is missing.
        """
        for attr in attr_path.split("."):
            obj = getattr(obj, attr, default)
            if obj is default:
                return default
        return obj

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

    @property
    def task(self) -> TaskType:
        """Get the task type of the datamodule."""
        if hasattr(self, "train_data"):
            return self.train_data.task
        if hasattr(self, "val_data"):
            return self.val_data.task
        if hasattr(self, "test_data"):
            return self.test_data.task
        msg = "This datamodule does not have any datasets. Did you call setup?"
        raise AttributeError(msg)

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
            self.val_data = copy.deepcopy(self.test_data)
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
