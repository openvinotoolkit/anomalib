from itertools import chain
from pathlib import Path
from collections import OrderedDict

import pytest
import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch

from anomalib.models.components import AnomalyModule
from anomalib.utils.callbacks.metrics_configuration import MetricsConfigurationCallback
from anomalib.utils.metrics.collection import AnomalibMetricCollection
from tests.helpers.dummy import DummyDataModule, DummyLogger, DummyModel


class _DummyAnomalyModule(AnomalyModule):
    def __init__(self):
        super().__init__()
        self.model = DummyModel()
        self.task = "segmentation"
        self.mode = "full"
        self.callbacks = []

    def test_step(self, batch, _):
        return None

    def validation_epoch_end(self, outputs):
        return None

    def test_epoch_end(self, outputs):
        return None

    def configure_optimizers(self):
        return None


@pytest.fixture
def config_from_yaml(request):
    return OmegaConf.load(Path(__file__).parent / request.param)


@pytest.mark.parametrize(
    ["config_from_yaml"],
    [("data/config-good-00.yaml",), ("data/config-good-01.yaml",)],
    indirect=["config_from_yaml"],
)
def test_metric_collection_configuration_callback(config_from_yaml):
    """Test if metrics are properly instantiated."""

    callback = MetricsConfigurationCallback(
        task="segmentation", image_metrics=config_from_yaml.metrics.image, pixel_metrics=config_from_yaml.metrics.pixel
    )

    dummy_logger = DummyLogger()
    dummy_anomaly_module = _DummyAnomalyModule()
    trainer = pl.Trainer(
        callbacks=[callback], logger=dummy_logger, enable_checkpointing=False, default_root_dir=dummy_logger.tempdir
    )
    callback.setup(trainer, dummy_anomaly_module, DummyDataModule())

    assert isinstance(
        dummy_anomaly_module.image_metrics, AnomalibMetricCollection
    ), f"{dummy_anomaly_module.image_metrics}"
    assert isinstance(
        dummy_anomaly_module.pixel_metrics, AnomalibMetricCollection
    ), f"{dummy_anomaly_module.pixel_metrics}"


@pytest.mark.parametrize(
    ["ori_config_from_yaml", "saved_config_from_yaml"],
    [("data/config-good-02.yaml", "data/config-good-02-serialized.yaml")],
)
def test_metric_collection_configuration_deserialzation_callback(ori_config_from_yaml, saved_config_from_yaml):
    """Test if metrics are properly instantiated during deserialzation."""

    ori_config_from_yaml_res = OmegaConf.load(Path(__file__).parent / ori_config_from_yaml)
    saved_config_from_yaml_res = OmegaConf.load(Path(__file__).parent / saved_config_from_yaml)
    callback = MetricsConfigurationCallback(
        task="segmentation",
        image_metrics=ori_config_from_yaml_res.metrics.image,
        pixel_metrics=ori_config_from_yaml_res.metrics.pixel,
    )

    dummy_logger = DummyLogger()
    dummy_anomaly_module = _DummyAnomalyModule()
    trainer = pl.Trainer(
        callbacks=[callback], logger=dummy_logger, enable_checkpointing=False, default_root_dir=dummy_logger.tempdir
    )

    saved_image_state_dict = OrderedDict(
        {
            "image_metrics." + k: torch.tensor(1.0)
            for k, v in saved_config_from_yaml_res.metrics.image.items()
            if v["class_path"].startswith("anomalib.utils.metrics")
        }
    )
    saved_pixel_state_dict = OrderedDict(
        {
            "pixel_metrics." + k: torch.tensor(1.0)
            for k, v in saved_config_from_yaml_res.metrics.pixel.items()
            if v["class_path"].startswith("anomalib.utils.metrics")
        }
    )

    final_state_dict = OrderedDict(chain(saved_image_state_dict.items(), saved_pixel_state_dict.items()))

    dummy_anomaly_module._load_metrics(final_state_dict)
    callback.setup(trainer, dummy_anomaly_module, DummyDataModule())

    assert isinstance(
        dummy_anomaly_module.image_metrics, AnomalibMetricCollection
    ), f"{dummy_anomaly_module.image_metrics}"
    assert isinstance(
        dummy_anomaly_module.pixel_metrics, AnomalibMetricCollection
    ), f"{dummy_anomaly_module.pixel_metrics}"

    assert sorted((list(dummy_anomaly_module.pixel_metrics))) == ["AUPRO", "AUROC", "F1Score"]
