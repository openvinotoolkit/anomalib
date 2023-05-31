from pathlib import Path

import pytest
from omegaconf import OmegaConf

from anomalib.trainer import AnomalibTrainer
from anomalib.utils.metrics.collection import AnomalibMetricCollection


@pytest.fixture
def config_from_yaml(request):
    return OmegaConf.load(Path(__file__).parent / request.param)


@pytest.mark.parametrize(
    ["config_from_yaml"],
    [("data/config-good-00.yaml",), ("data/config-good-01.yaml",)],
    indirect=["config_from_yaml"],
)
def test_metrics_initialization(config_from_yaml):
    """Test if metrics are properly instantiated."""

    trainer = AnomalibTrainer(
        logger=False, image_metrics=config_from_yaml.metrics.image, pixel_metrics=config_from_yaml.metrics.pixel
    )
    trainer.metrics_connector.initialize()

    assert isinstance(
        trainer.metrics_connector.image_metrics, AnomalibMetricCollection
    ), f"{trainer.metrics_connector.image_metrics}"
    assert isinstance(
        trainer.metrics_connector.pixel_metrics, AnomalibMetricCollection
    ), f"{trainer.metrics_connector.pixel_metrics}"
