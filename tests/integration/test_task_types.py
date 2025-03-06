"""Tests to check behaviour of the auxiliary components across different task types (classification, segmentation) ."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
from torchmetrics import Metric

from anomalib import LearningType
from anomalib.data import AnomalibDataModule, Batch, Folder, ImageDataFormat
from anomalib.engine import Engine
from anomalib.metrics import AnomalibMetric, Evaluator
from anomalib.models import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.visualization import ImageVisualizer
from tests.helpers.data import DummyImageDatasetGenerator


class DummyBaseModel(AnomalibModule):
    """Dummy model for testing.

    No training, and all auxiliary components default to None. This allows testing of the different components
    in isolation.
    """

    def training_step(self, *args, **kwargs) -> None:
        """Dummy training step."""

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Run for single epoch."""
        return {"max_epochs": 1}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model."""
        return LearningType.ONE_CLASS

    def configure_optimizers(self) -> None:
        """No optimizers needed."""

    def configure_preprocessor(self) -> None:
        """No default pre-processor needed."""

    def configure_post_processor(self) -> None:
        """No default post-processor needed."""

    def configure_evaluator(self) -> None:
        """No default evaluator needed."""

    def configure_visualizer(self) -> None:
        """No default visualizer needed."""


class DummyClassificationModel(DummyBaseModel):
    """Dummy classification model for testing.

    Validation step returns random image-only scores, to simulate a model that performs classification.
    """

    def validation_step(self, batch: Batch, *args, **kwargs) -> Batch:
        """Validation steps that returns random image-level scores."""
        del args, kwargs
        batch.pred_score = torch.rand(batch.batch_size, device=self.device)
        return batch


class DummySegmentationModel(DummyBaseModel):
    """Dummy segmentation model for testing.

    Validation step returns random image- and pixel-level scores, to simulate a model that performs segmentation.
    """

    def validation_step(self, batch: Batch, *args, **kwargs) -> Batch:
        """Validation steps that returns random image- and pixel-level scores."""
        del args, kwargs
        batch.pred_score = torch.rand(batch.batch_size, device=self.device)
        batch.anomaly_map = torch.rand(batch.batch_size, *batch.image.shape[-2:], device=self.device)
        return batch


class _DummyMetric(Metric):
    """Dummy metric for testing."""

    def update(self, *args, **kwargs) -> None:
        """Dummy update method."""

    def compute(self) -> None:
        """Dummy compute method."""
        assert self.update_called  # simulate failure to compute if states are not updated


class DummyMetric(AnomalibMetric, _DummyMetric):
    """Dummy Anomalib metric for testing."""


@pytest.fixture
def folder_dataset_path(project_path: Path) -> Path:
    """Create a dummy folder dataset for testing."""
    data_path = project_path / "dataset"
    dataset_generator = DummyImageDatasetGenerator(
        data_format=ImageDataFormat.FOLDER,
        root=data_path,
        num_train=10,
        num_test=10,
    )
    dataset_generator.generate_dataset()
    return data_path


@pytest.fixture
def classification_datamodule(folder_dataset_path: Path) -> AnomalibDataModule:
    """Create a classification datamodule for testing.

    The datamodule is created with a folder dataset, that does not have a mask directory.
    """
    # create the folder datamodule
    return Folder(
        name="cls_dataset",
        root=folder_dataset_path,
        normal_dir="good",
        abnormal_dir="bad",
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
    )


@pytest.fixture
def segmentation_datamodule(folder_dataset_path: Path) -> AnomalibDataModule:
    """Create a segmentation datamodule for testing.

    The datamodule is created with a folder dataset, that has a mask directory.
    """
    # create the folder datamodule
    return Folder(
        name="seg_dataset",
        root=folder_dataset_path,
        normal_dir="good",
        abnormal_dir="bad",
        mask_dir="masks",  # include masks for segmentation dataset
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
    )


@pytest.fixture
def image_and_pixel_evaluator() -> Evaluator:
    """Create an evaluator with image- and pixel-level metrics for testing."""
    image_metric = DummyMetric(fields=["pred_score", "gt_label"], prefix="image_")
    pixel_metric = DummyMetric(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False)
    val_metrics = [image_metric, pixel_metric]
    test_metrics = copy.deepcopy(val_metrics)
    return Evaluator(val_metrics=[image_metric, pixel_metric], test_metrics=test_metrics)


@pytest.fixture
def engine(project_path: Path) -> Engine:
    """Create an engine for testing.

    Run on cpu to speed up tests.
    """
    return Engine(accelerator="cpu", default_root_dir=project_path)


class TestEvaluation:
    """Test evaluation across task types.

    Tests if image- and/or pixel- metrics are computed without errors for models and datasets with different task types.
    """

    @staticmethod
    def test_cls_model_cls_dataset(
        engine: Engine,
        classification_datamodule: AnomalibDataModule,
        image_and_pixel_evaluator: Evaluator,
    ) -> None:
        """Test classification model with classification dataset."""
        model = DummyClassificationModel(evaluator=image_and_pixel_evaluator)
        engine.train(model, datamodule=classification_datamodule)

    @staticmethod
    def test_cls_model_seg_dataset(
        engine: Engine,
        segmentation_datamodule: AnomalibDataModule,
        image_and_pixel_evaluator: Evaluator,
    ) -> None:
        """Test classification model with segmentation dataset."""
        model = DummyClassificationModel(evaluator=image_and_pixel_evaluator)
        engine.train(model, datamodule=segmentation_datamodule)

    @staticmethod
    def test_seg_model_cls_dataset(
        engine: Engine,
        classification_datamodule: AnomalibDataModule,
        image_and_pixel_evaluator: Evaluator,
    ) -> None:
        """Test segmentation model with classification dataset."""
        model = DummySegmentationModel(evaluator=image_and_pixel_evaluator)
        engine.train(model, datamodule=classification_datamodule)

    @staticmethod
    def test_seg_model_seg_dataset(
        engine: Engine,
        segmentation_datamodule: AnomalibDataModule,
        image_and_pixel_evaluator: Evaluator,
    ) -> None:
        """Test segmentation model with segmentation dataset."""
        model = DummySegmentationModel(evaluator=image_and_pixel_evaluator)
        engine.train(model, datamodule=segmentation_datamodule)


class TestPostProcessing:
    """Tests post-processing across task types.

    Tests if post-processing is applied without errors for models and datasets with different task types.
    """

    @staticmethod
    def test_cls_model_cls_dataset(engine: Engine, classification_datamodule: AnomalibDataModule) -> None:
        """Test classification model with classification dataset."""
        model = DummyClassificationModel(post_processor=PostProcessor())
        engine.train(model, datamodule=classification_datamodule)

    @staticmethod
    def test_cls_model_seg_dataset(engine: Engine, segmentation_datamodule: AnomalibDataModule) -> None:
        """Test classification model with segmentation dataset."""
        model = DummyClassificationModel(post_processor=PostProcessor())
        engine.train(model, datamodule=segmentation_datamodule)

    @staticmethod
    def test_seg_model_cls_dataset(engine: Engine, classification_datamodule: AnomalibDataModule) -> None:
        """Test segmentation model with classification dataset."""
        model = DummySegmentationModel(post_processor=PostProcessor())
        engine.train(model, datamodule=classification_datamodule)

    @staticmethod
    def test_seg_model_seg_dataset(engine: Engine, segmentation_datamodule: AnomalibDataModule) -> None:
        """Test segmentation model with segmentation dataset."""
        model = DummySegmentationModel(post_processor=PostProcessor())
        engine.train(model, datamodule=segmentation_datamodule)


class TestVisualization:
    """Tests visualization across task types.

    Tests if visualizations are created without errors for models and datasets with different task types.
    """

    @staticmethod
    def test_cls_model_cls_dataset(engine: Engine, classification_datamodule: AnomalibDataModule) -> None:
        """Test classification model with classification dataset."""
        model = DummyClassificationModel(visualizer=ImageVisualizer())
        engine.train(model, datamodule=classification_datamodule)

    @staticmethod
    def test_cls_model_seg_dataset(engine: Engine, segmentation_datamodule: AnomalibDataModule) -> None:
        """Test classification model with segmentation dataset."""
        model = DummyClassificationModel(visualizer=ImageVisualizer())
        engine.train(model, datamodule=segmentation_datamodule)

    @staticmethod
    def test_seg_model_cls_dataset(engine: Engine, classification_datamodule: AnomalibDataModule) -> None:
        """Test segmentation model with classification dataset."""
        model = DummySegmentationModel(visualizer=ImageVisualizer())
        engine.train(model, datamodule=classification_datamodule)

    @staticmethod
    def test_seg_model_seg_dataset(engine: Engine, segmentation_datamodule: AnomalibDataModule) -> None:
        """Test segmentation model with segmentation dataset."""
        model = DummySegmentationModel(visualizer=ImageVisualizer())
        engine.train(model, datamodule=segmentation_datamodule)
