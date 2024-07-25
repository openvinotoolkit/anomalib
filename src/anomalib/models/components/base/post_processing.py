
from lightning import LightningModule, Trainer
import torch
from torch import nn
from lightning.pytorch import Callback
from anomalib.metrics import MinMax, F1AdaptiveThreshold

from dataclasses import replace
from anomalib.dataclasses import Batch, InferenceBatch
from abc import ABC, abstractmethod


class PostProcessor(nn.Module, Callback, ABC):

    @abstractmethod
    def forward(self, pred_score, anomaly_map):
        """ Funcional forward method for post-processing.
        """
        pass

    @abstractmethod
    def post_process_batch(self, batch: Batch):
        """ Post-process the predictions.
        """
        pass



class OneClassPostProcessor(PostProcessor):
    """ Default post-processor for one-class anomaly detection.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._image_threshold = F1AdaptiveThreshold()
        self._pixel_threshold = F1AdaptiveThreshold()
        self._normalization_stats = MinMax()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Batch,
        *args,
        **kwargs,
    ) -> None:
        del trainer, pl_module # Unused arguments.
        self._image_threshold.update(outputs.pred_score, outputs.gt_label)
        self._pixel_threshold.update(outputs.anomaly_map, outputs.gt_mask)
        self._normalization_stats.update(outputs.anomaly_map)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._image_threshold.compute()
        self._pixel_threshold.compute()
        self._normalization_stats.compute()

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Batch, *args, **kwargs) -> None:
        del trainer, pl_module
        self.post_process_batch(outputs)

    def on_predict_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Batch, *args, **kwargs) -> None:
        del trainer, pl_module
        self.post_process_batch(outputs)

    def forward(self, predictions: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        """ Funcional forward method for post-processing.
        """
        if isinstance(predictions, tuple):
            pred_score, anomaly_map = predictions
            pred_score = self._normalize(pred_score, self.min, self.max, self.image_threshold)
        else:
            anomaly_map = predictions
            pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        pred_label = self._threshold(pred_score, self.image_threshold)
        pred_mask = self._threshold(anomaly_map, self.pixel_threshold)
        pred_score = self._normalize(pred_score, self.min, self.max, self.image_threshold)
        anomaly_map = self._normalize(anomaly_map, self.min, self.max, self.pixel_threshold)
        return InferenceBatch(
            pred_label=pred_label,
            pred_score=pred_score,
            pred_mask=pred_mask,
            anomaly_map=anomaly_map
        )

    def post_process_batch(self, batch: Batch):
        # apply threshold
        self.threshold_batch(batch)
        # apply normalization
        self.normalize_batch(batch)

    def threshold_batch(self, batch: Batch):
        batch.pred_label = batch.pred_label or self._threshold(batch.pred_score, self.image_threshold)
        batch.pred_mask = batch.pred_mask or self._threshold(batch.anomaly_map, self.pixel_threshold)

    def normalize_batch(self, batch: Batch):
        # normalize image-level predictions
        batch.pred_score = self._normalize(batch.pred_score, self.min, self.max, self.image_threshold)
        # normalize pixel-level predictions
        batch.anomaly_map = self._normalize(batch.anomaly_map, self.min, self.max, self.pixel_threshold)

    @staticmethod
    def _threshold(preds, threshold):
        return preds > threshold

    @staticmethod
    def _normalize(preds, min, max, threshold):
        preds = ((preds - threshold) / (max - min)) + 0.5
        preds = torch.minimum(preds, torch.tensor(1))
        preds = torch.maximum(preds, torch.tensor(0))
        return preds

    @property
    def image_threshold(self):
        return self._image_threshold.value
    
    @property
    def pixel_threshold(self):
        return self._pixel_threshold.value

    @property
    def min(self):
        return self._normalization_stats.min
    
    @property
    def max(self):
        return self._normalization_stats.max
