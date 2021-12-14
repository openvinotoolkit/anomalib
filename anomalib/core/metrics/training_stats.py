"""Module that computes the parameters of the normal data distribution of the training set."""
from typing import Dict, Optional

import torch
from torch import Tensor
from torchmetrics import Metric


class TrainingStats(Metric):
    """Mean and standard deviation of the anomaly scores of normal training data."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.anomaly_maps = []
        self.anomaly_scores = []

        self.image_mean = Tensor(0)
        self.image_std = Tensor(0)
        self.pixel_mean = Tensor(0)
        self.pixel_std = Tensor(0)

        self.persistent(mode=True)

    # pylint: disable=arguments-differ
    def update(
        self, anomaly_maps: Optional[Tensor] = None, anomaly_scores: Optional[Tensor] = None
    ) -> None:  # type: ignore
        """Update the precision-recall curve metric."""
        if anomaly_maps is not None:
            self.anomaly_maps.append(anomaly_maps)
        if anomaly_scores is not None:
            self.anomaly_scores.append(anomaly_scores)

    def compute(self) -> Dict[str, Tensor]:
        """Compute stats."""
        anomaly_maps = torch.vstack(self.anomaly_maps)
        anomaly_maps = torch.log(anomaly_maps).cpu()
        if len(self.anomaly_scores) == 0:
            anomaly_scores = anomaly_maps.reshape(anomaly_maps.shape[0], -1).max(axis=1).values
        else:
            anomaly_scores = torch.hstack(self.anomaly_scores)

        self.image_mean = anomaly_scores.mean()
        self.image_std = anomaly_scores.std()

        # per pixel stats
        self.pixel_mean = anomaly_maps.mean(dim=0).squeeze()
        self.pixel_std = anomaly_maps.std(dim=0).squeeze()

        metrics = dict(
            image_mean=self.image_mean, image_std=self.image_std, pixel_mean=self.pixel_mean, pixel_std=self.pixel_std
        )
        del self.anomaly_scores
        del self.anomaly_maps
        return metrics
