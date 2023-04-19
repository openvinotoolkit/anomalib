"""Tests if the AnomalibTrainer produces the right results."""

import pytest
import torch

from anomalib.trainer import AnomalibTrainer
from anomalib.utils.metrics.min_max import MinMax
from tests.helpers.dataloaders.tensor import DummyTensorDataModule

from .dummy_model import DummyAnomalibModule


class TestAnomalibTrainer:
    # TODO test both min max and cdf
    @pytest.fixture(scope="class")
    def trained_model(self):
        model = DummyAnomalibModule()
        datamodule = DummyTensorDataModule()
        trainer = AnomalibTrainer(
            logger=False, max_epochs=2, image_metrics=["F1Score"], pixel_metrics=["F1Score"], num_sanity_val_steps=0
        )
        trainer.fit(model=model, datamodule=datamodule)
        return trainer, model, datamodule

    @pytest.mark.parametrize("stage", ["predict", "validate", "test"])
    def test_stage(self, trained_model, stage):
        trainer, model, datamodule = trained_model
        if stage == "predict":
            outputs = trainer.predict(model, datamodule)
        elif stage == "validate":
            outputs = trainer.validate(model, datamodule)
        elif stage == "test":
            outputs = trainer.test(model, datamodule)

        if stage == "predict":
            output = outputs[0]
            anomaly_map = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
            anomaly_map[:, :, 5:15, 5:15] = 0.5
            assert (output["anomaly_maps"] == anomaly_map).all()

            pred_mask = torch.zeros((32, 32), dtype=torch.int64)
            pred_mask[5:15, 5:15] = 1
            assert (output["pred_masks"] == pred_mask).all()

            assert model.image_threshold.value == 3.0
            assert model.pixel_threshold.value == 3.0
            assert isinstance(model.normalization_metrics, MinMax)
        elif stage == "test":
            assert "image_F1Score" in outputs[0].keys()
            assert "pixel_F1Score" in outputs[0].keys()
            assert outputs[0]["image_F1Score"] == 1.0
            assert outputs[0]["pixel_F1Score"] == 1.0
        else:
            assert model.image_threshold.value == 3.0
            assert model.pixel_threshold.value == 3.0
            assert isinstance(model.normalization_metrics, MinMax)
