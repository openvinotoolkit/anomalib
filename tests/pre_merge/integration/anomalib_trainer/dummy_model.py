"""Dummy model to test the AnomalibTrainer."""

from einops import reduce

from anomalib.models import AnomalyModule
from anomalib.post_processing.post_process import ThresholdMethod
from anomalib.utils.metrics import create_metric_collection
from anomalib.utils.metrics.min_max import MinMax


class DummyAnomalibModule(AnomalyModule):
    def training_step(self, *args, **kwargs):
        pass

    def training_epoch_end(self, outputs) -> None:
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, *args, **kwargs):
        batch["anomaly_maps"] = reduce(batch["images"], "b c h w -> b 1 h w", "sum")
        return batch

    def test_step(self, batch, *args, **kwargs):
        return self.validation_step(batch, *args, **kwargs)

    def predict_step(self, batch, *args, **kwargs):
        return self.validation_step(batch, *args, **kwargs)

    def configure_optimizers(self):
        return None
