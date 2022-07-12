"""Implementation of AUPRO score based on TorchMetrics."""
from typing import Any, Callable, List, Optional, Tuple

import torch
from kornia.contrib import connected_components
from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import auc, roc
from torchmetrics.utilities.data import dim_zero_cat

from .plotting_utils import plot_figure


class AUPRO(Metric):
    """Area under per region overlap (AUPRO) Metric."""

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        fpr_limit: float = 0.3,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("preds", default=[], dist_reduce_fx="cat")  # pylint: disable=not-callable
        self.add_state("target", default=[], dist_reduce_fx="cat")  # pylint: disable=not-callable
        self.fpr_limit = fpr_limit

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with new values.

        Args:
            preds (Tensor): predictions of the model
            target (Tensor): ground truth targets
        """
        self.target.append(target)
        self.preds.append(preds)

    def _compute(self) -> Tuple[Tensor, Tensor]:
        """Compute the pro/fpr value-pairs until the fpr specified by self.fpr_limit.

        It leverages the fact that the overlap corresponds to the tpr, and thus computes the overall
        tpr/pro by aggregating per-region tprs produced by ROC-construction.

        Raises:
            ValueError: ValueError is raised if self.target doesn't conform with requirements imposed by kornia for
                        connected component analysis.

        Returns:
            Tuple[Tensor, Tensor]: tuple containing final fpr and pro values.
        """
        target = dim_zero_cat(self.target)
        preds = dim_zero_cat(self.preds)

        # check and prepare target for labeling via kornia
        if target.min() < 0 or target.max() > 1:
            raise ValueError(
                (
                    f"kornia.contrib.connected_components expects input to lie in the interval [0, 1], but found "
                    f"interval was [{target.min()}, {target.max()}]."
                )
            )
        target = target.unsqueeze(1)  # kornia expects N1HW format
        target = target.type(torch.float)  # kornia expects FloatTensor
        cca = connected_components(target)

        preds = preds.flatten()
        cca = cca.flatten()
        target = target.flatten()

        # compute the global fpr and extract correspondig idx
        fpr, _, _ = roc(preds, target)
        fpr_idx = torch.where(fpr <= self.fpr_limit)[0]
        fpr = fpr[fpr_idx]

        # compute the pro value by aggregating per-region tpr values.
        pro = torch.zeros_like(fpr_idx, device=preds.device, dtype=torch.float)

        labels = cca.unique()[1:]  # 0 is background
        for label in labels:
            mask = cca == label
            _, tpr, _ = roc(preds, mask)
            pro += tpr[fpr_idx]

        pro /= labels.size(0)
        return fpr, pro

    def compute(self) -> Tensor:
        """Fist compute PRO curve, then compute and scale area under the curve.

        Returns:
            Tensor: Value of the AUPRO metric
        """
        fpr, pro = self._compute()

        aupro = auc(fpr, pro)
        aupro = aupro / fpr[-1]  # normalize the area

        return aupro

    def generate_figure(self) -> Tuple[Figure, str]:
        """Generate a figure containing the PRO curve and the AUPRO.

        Returns:
            Tuple[Figure, str]: Tuple containing both the figure and the figure title to be used for logging
        """
        fpr, pro = self._compute()
        aupro = self.compute()

        xlim = (0.0, self.fpr_limit)
        ylim = (0.0, 1.0)
        xlabel = "False Positive Rate"
        ylabel = "Per-Region Overlap/TPR"
        loc = "lower right"
        title = "PRO"

        fig, _axis = plot_figure(fpr, pro, aupro, xlim, ylim, xlabel, ylabel, loc, title)

        return fig, "PRO"
