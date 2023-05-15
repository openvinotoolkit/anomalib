"""Implements custom trainer for Anomalib."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings
from typing import List, Optional

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.types import _EVALUATE_OUTPUT, EVAL_DATALOADERS, TRAIN_DATALOADERS

from anomalib.data import AnomalibDataModule, AnomalibDataset, TaskType
from anomalib.models.components.base.anomaly_module import AnomalyModule
from anomalib.post_processing import NormalizationMethod, ThresholdMethod, VisualizationMode
from anomalib.trainer.loops.one_class import FitLoop, PredictionLoop, TestLoop, ValidationLoop
from anomalib.trainer.utils import (
    CheckpointConnector,
    MetricsManager,
    PostProcessor,
    Thresholder,
    VisualizationManager,
    VisualizationStage,
    get_normalizer,
)
from anomalib.utils.metrics import AnomalyScoreThreshold

from .utils import CallbackConnector

log = logging.getLogger(__name__)
# warnings to ignore in trainer
warnings.filterwarnings(
    "ignore", message="torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead"
)


class AnomalibTrainer(Trainer):
    """Anomalib trainer.

    Note:
        Refer to PyTorch Lightning's Trainer for a list of parameters for details on other Trainer parameters.

    Args:
        threshold_method (ThresholdMethod): Thresholding method for normalizer.
        normalization_method (NormalizationMethod): Normalization method
        manual_image_threshold (Optional[float]): If threshold method is manual, this needs to be set. Defaults to None.
        manual_pixel_threshold (Optional[float]): If threshold method is manual, this needs to be set. Defaults to None.
        visualization_mode (str): Visualization mode. Options ["full", "simple"]. Defaults to "full".
        show_images (bool): Whether to show images. Defaults to False.
        log_images (bool): Whether to log images. Defaults to False.
        visualization_stage (VisualizationStage): The stage at which to write images to the logger(s).
            Defaults to VisualizationStage.TEST.
    """

    def __init__(
        self,
        threshold_method: ThresholdMethod = ThresholdMethod.ADAPTIVE,
        normalization_method: NormalizationMethod = NormalizationMethod.MIN_MAX,
        manual_image_threshold: Optional[float] = None,
        manual_pixel_threshold: Optional[float] = None,
        image_metrics: Optional[List[str]] = None,
        pixel_metrics: Optional[List[str]] = None,
        visualization_mode: VisualizationMode = VisualizationMode.FULL,
        show_images: bool = False,
        log_images: bool = False,
        visualization_stage: VisualizationStage = VisualizationStage.TEST,
        task_type: TaskType = TaskType.SEGMENTATION,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._checkpoint_connector = CheckpointConnector(self, kwargs.get("resume_from_checkpoint", None))

        self._callback_connector = CallbackConnector(self)
        self._checkpoint_connector = CheckpointConnector(self, kwargs.get("resume_from_checkpoint", None))

        self.lightning_module: AnomalyModule  # for mypy

        self.fit_loop = FitLoop(min_epochs=kwargs.get("min_epochs", 0), max_epochs=kwargs.get("max_epochs", None))
        self.validate_loop = ValidationLoop()
        self.test_loop = TestLoop()
        self.predict_loop = PredictionLoop()

        self._callback_connector.on_trainer_init(
            callbacks=kwargs.get("callbacks", None),
            enable_checkpointing=kwargs.get("enable_checkpointing", True),
            enable_progress_bar=kwargs.get("enable_progress_bar", True),
            default_root_dir=kwargs.get("default_root_dir"),
            enable_model_summary=kwargs.get("enable_model_summary", True),
            max_time=kwargs.get("max_time", None),
            accumulate_grad_batches=kwargs.get("accumulate_grad_batches", None),
        )

        self.task_type = task_type
        # these are part of the trainer as they are used in the metrics-manager, post-processor and thresholder
        self.image_threshold = AnomalyScoreThreshold().cpu()
        self.pixel_threshold = AnomalyScoreThreshold().cpu()

        self.thresholder = Thresholder(
            trainer=self,
            threshold_method=threshold_method,
            manual_image_threshold=manual_image_threshold,
            manual_pixel_threshold=manual_pixel_threshold,
        )
        self.post_processor = PostProcessor(trainer=self)
        self.normalizer = get_normalizer(trainer=self, normalization_method=normalization_method)
        self.metrics = MetricsManager(trainer=self, image_metrics=image_metrics, pixel_metrics=pixel_metrics)
        self.visualization_manager = VisualizationManager(
            trainer=self,
            mode=visualization_mode,
            show_images=show_images,
            log_images=log_images,
            stage=visualization_stage,
        )

    def fit(
        self,
        model: LightningModule,
        train_dataloaders: TRAIN_DATALOADERS | LightningDataModule | None = None,
        val_dataloaders: EVAL_DATALOADERS | None = None,
        datamodule: LightningDataModule | None = None,
        ckpt_path: str | None = None,
    ) -> None:
        """Sets task type in the dataset and calls the fit method of the trainer."""
        if datamodule is not None:
            self._set_datamodule_task(datamodule)

        if train_dataloaders is not None:
            self._set_dataloader_task(train_dataloaders)

        if val_dataloaders is not None:
            self._set_dataloader_task(val_dataloaders)

        return super().fit(model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)

    def test(
        self,
        model: LightningModule | None = None,
        dataloaders: EVAL_DATALOADERS | LightningDataModule | None = None,
        ckpt_path: str | None = None,
        verbose: bool = True,
        datamodule: LightningDataModule | None = None,
    ) -> _EVALUATE_OUTPUT:
        """Sets the task type in the dataset and calls the test method of the trainer."""
        if datamodule is not None:
            self._set_datamodule_task(datamodule)

        if dataloaders is not None:
            self._set_dataloader_task(dataloaders)

        return super().test(model, dataloaders, ckpt_path, verbose, datamodule)

    def _set_datamodule_task(self, datamodule: LightningDataModule):
        """Sets task type parameter in the dataset of the datamodule.

        Args:
            datamodule (LightningDataModule): Lightning datamodule.
        """
        if isinstance(datamodule, AnomalibDataModule):
            for attribute in ("val_data", "test_data", "train_data"):
                if hasattr(datamodule, attribute):
                    dataset = getattr(datamodule, attribute)
                    if isinstance(dataset, AnomalibDataset):
                        dataset.task = self.task_type

    def _set_dataloader_task(self, dataloader: TRAIN_DATALOADERS | EVAL_DATALOADERS | LightningDataModule):
        """Sets task type parameter in the dataset of the dataloader or datamodule.

        Args:
            dataloader (TRAIN_DATALOADERS | EVAL_DATALOADERS | LightningDataModule): Lightning dataloader.
        """
        if isinstance(dataloader, LightningDataModule):
            self._set_datamodule_task(dataloader)
        elif isinstance(dataloader, (list, tuple)):
            for dataloader in dataloader:
                if isinstance(dataloader, LightningDataModule):
                    self._set_datamodule_task(dataloader)
        elif isinstance(dataloader, dict):
            for dataloader in dataloader.values():
                if isinstance(dataloader, LightningDataModule):
                    self._set_datamodule_task(dataloader)
        else:
            warnings.warn(f"train_dataloaders is of type {type(dataloader)}. Skipping setting task type.")
