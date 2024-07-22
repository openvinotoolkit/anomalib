
import torch
from torch import nn
from torchmetrics import MetricCollection
from anomalib.metrics import MinMax, F1AdaptiveThreshold

from dataclasses import replace
from anomalib.dataclasses import PredictBatch
from abc import ABC, abstractmethod


class PostProcessor(nn.Module, ABC):

    @abstractmethod
    def update(self, batch: PredictBatch):
        """ Update the min and max values.
        """
        pass

    @abstractmethod
    def forward(self, pred_score, anomaly_map):
        """ Funcional forward method for post-processing.
        """
        pass

    @abstractmethod
    def post_process_batch(self, batch: PredictBatch):
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

    def update(self, batch: PredictBatch):
        """ Update the min and max values.
        """
        self._image_threshold.update(batch.pred_score, batch.gt_label)
        self._pixel_threshold.update(batch.anomaly_map, batch.gt_mask)
        self._normalization_stats.update(batch.anomaly_map)

    def compute(self):
        """ Compute the min and max values.
        """
        self._image_threshold.compute()
        self._pixel_threshold.compute()
        self._normalization_stats.compute()

    def forward(self, pred_score: torch.Tensor, anomaly_map):
        """ Funcional forward method for post-processing.
        """
        pred_score = self._normalize(pred_score, self.min, self.max, self.image_threshold)
        anomaly_map = self._normalize(anomaly_map, self.min, self.max, self.pixel_threshold)
        return pred_score, anomaly_map

    def post_process_batch(self, batch: PredictBatch):
        # apply threshold
        batch = self.threshold_batch(batch)
        # apply normalization
        return self.normalize_batch(batch)

    def threshold_batch(self, batch: PredictBatch):
        pred_label = self._threshold(batch.pred_score, self.image_threshold)
        pred_mask = self._threshold(batch.anomaly_map, self.pixel_threshold)
        return replace(batch, pred_label=pred_label, pred_mask=pred_mask)

    def normalize_batch(self, batch: PredictBatch):
        # normalize image-level predictions
        pred_score = self._normalize(batch.pred_score, self.min, self.max, self.image_threshold)
        # normalize pixel-level predictions
        anomaly_map = self._normalize(batch.anomaly_map, self.min, self.max, self.pixel_threshold)
        return replace(batch, pred_score=pred_score, anomaly_map=anomaly_map)

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
