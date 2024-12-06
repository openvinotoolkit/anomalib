"""Test metrics callback."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from itertools import chain
from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms.v2 import Resize

from anomalib import LearningType
from anomalib.callbacks.metrics import _MetricsCallback
from anomalib.metrics import AnomalibMetricCollection
from anomalib.metrics.threshold import F1AdaptiveThreshold
from anomalib.models.components import AnomalyModule


class _DummyAnomalyModule(AnomalyModule):
    def __init__(self) -> None:
        super().__init__()
        self.task = "segmentation"
        self.mode = "full"
        self.callbacks = []
        self.image_threshold = F1AdaptiveThreshold()
        self.pixel_threshold = F1AdaptiveThreshold()

    @staticmethod
    def test_step(**_kwdargs) -> None:
        return None

    @staticmethod
    def validation_epoch_end(**_kwdargs) -> None:
        return None

    @staticmethod
    def test_epoch_end(**_kwdargs) -> None:
        return None

    @staticmethod
    def configure_optimizers() -> None:
        return None

    @property
    def learning_type(self) -> LearningType:
        """Learning type of the model."""
        return LearningType.ONE_CLASS

    @property
    def trainer_arguments(self) -> dict:
        return {}

    @property
    def configure_transforms(self) -> None:
        return Resize((256, 256))


@pytest.fixture()
def config_from_yaml(request: "pytest.FixtureRequest") -> DictConfig:
    """Loads config from path."""
    return OmegaConf.load(Path(__file__).parent / request.param)


@pytest.mark.parametrize(
    "config_from_yaml",
    ["data/config-good-00.yaml", "data/config-good-01.yaml"],
    indirect=["config_from_yaml"],
)
def test_metric_collection_configuration_callback(config_from_yaml: str, tmpdir: str) -> None:
    """Test if metrics are properly instantiated."""
    callback = _MetricsCallback(
        task="segmentation",
        image_metrics=config_from_yaml.metrics.image,
        pixel_metrics=config_from_yaml.metrics.pixel,
    )

    dummy_anomaly_module = _DummyAnomalyModule()
    trainer = pl.Trainer(
        callbacks=[callback],
        enable_checkpointing=False,
        default_root_dir=tmpdir,
    )
    callback.setup(trainer, dummy_anomaly_module)

    assert isinstance(
        dummy_anomaly_module.image_metrics,
        AnomalibMetricCollection,
    ), f"{dummy_anomaly_module.image_metrics}"
    assert isinstance(
        dummy_anomaly_module.pixel_metrics,
        AnomalibMetricCollection,
    ), f"{dummy_anomaly_module.pixel_metrics}"


@pytest.mark.parametrize(
    ("ori_config_from_yaml", "saved_config_from_yaml"),
    [("data/config-good-02.yaml", "data/config-good-02-serialized.yaml")],
)
def test_metric_collection_configuration_deserialzation_callback(
    ori_config_from_yaml: str,
    saved_config_from_yaml: str,
    tmpdir: str,
) -> None:
    """Test if metrics are properly instantiated during deserialzation."""
    ori_config_from_yaml_res = OmegaConf.load(Path(__file__).parent / ori_config_from_yaml)
    saved_config_from_yaml_res = OmegaConf.load(Path(__file__).parent / saved_config_from_yaml)
    callback = _MetricsCallback(
        task="segmentation",
        image_metrics=ori_config_from_yaml_res.metrics.image,
        pixel_metrics=ori_config_from_yaml_res.metrics.pixel,
    )

    dummy_anomaly_module = _DummyAnomalyModule()
    trainer = pl.Trainer(
        callbacks=[callback],
        enable_checkpointing=False,
        default_root_dir=tmpdir,
    )

    saved_image_state_dict = OrderedDict(
        {
            "image_metrics." + k: torch.tensor(1.0)
            for k, v in saved_config_from_yaml_res.metrics.image.items()
            if v["class_path"].startswith("anomalib.metrics")
        },
    )
    saved_pixel_state_dict = OrderedDict(
        {
            "pixel_metrics." + k: torch.tensor(1.0)
            for k, v in saved_config_from_yaml_res.metrics.pixel.items()
            if v["class_path"].startswith("anomalib.metrics")
        },
    )

    final_state_dict = OrderedDict(chain(saved_image_state_dict.items(), saved_pixel_state_dict.items()))

    dummy_anomaly_module._load_metrics(final_state_dict)  # noqa: SLF001
    callback.setup(trainer, dummy_anomaly_module)

    assert isinstance(
        dummy_anomaly_module.image_metrics,
        AnomalibMetricCollection,
    ), f"{dummy_anomaly_module.image_metrics}"
    assert isinstance(
        dummy_anomaly_module.pixel_metrics,
        AnomalibMetricCollection,
    ), f"{dummy_anomaly_module.pixel_metrics}"

    for metric_name in ("AUROC", "F1Score"):
        assert metric_name in dummy_anomaly_module.pixel_metrics
