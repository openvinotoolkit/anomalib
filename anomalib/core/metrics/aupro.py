"""Implementation of AUPRO metric based on TorchMetrics."""
import warnings

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import auc, roc

from anomalib.core.metrics.pro import connected_components, pro_score


class AUPRO(Metric):
    """Area Under the Per-Region Overlap Curve (AUPRO)."""

    def __init__(self, auc_samples: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        if not torch.cuda.is_available():
            warnings.warn(
                "Computation of the PRO metric is optimized for the GPU, but cuda is not available on your device. "
                "Because of this, the PRO computation will significantly slow down code execution."
            )
        self.auc_samples = auc_samples
        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("comps", default=[], dist_reduce_fx="cat")
        self.add_state("n_regions", default=[], dist_reduce_fx="cat")

    def update(self, predictions: Tensor, targets: Tensor) -> None:  # type: ignore  # pylint: disable=arguments-differ
        """Compute the PRO score for the current batch."""
        self.predictions.append(predictions)
        self.targets.append(targets)
        if torch.cuda.is_available():
            targets = targets.cuda()
        comps, n_comps = connected_components(targets.unsqueeze(1))
        self.comps.append(comps.to(predictions.device))
        self.n_regions.append(n_comps - 1)

    def compute(self) -> Tensor:
        """Compute the macro average of the PRO score across all regions in all batches."""
        fpr, _, thresholds = roc(torch.vstack(self.predictions).flatten(), torch.vstack(self.targets).flatten())

        # downsample to speed up computation
        downsample_idx = torch.linspace(0, len(fpr) - 1, self.auc_samples).long()
        fpr = fpr[downsample_idx]
        thresholds = thresholds[downsample_idx]

        pro_scores = []
        for threshold in thresholds:
            pro = 0
            for preds, comps, n_regions in zip(self.predictions, self.comps, self.n_regions):
                if torch.cuda.is_available():
                    preds = preds.cuda()
                    comps = comps.cuda()
                pro += pro_score(preds, comps, threshold) * n_regions
            pro_scores.append(pro / sum(self.n_regions))
        return auc(fpr, Tensor(pro_scores))
