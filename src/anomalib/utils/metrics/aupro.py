"""Implementation of AUPRO score based on TorchMetrics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable

import torch
from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import auc, roc
from torchmetrics.utilities.data import dim_zero_cat

from anomalib.utils.metrics.pro import (
    connected_components_cpu,
    connected_components_gpu,
)

from .plotting_utils import plot_figure


class AUPRO(Metric):
    """Area under per region overlap (AUPRO) Metric."""

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False
    preds: list[Tensor]
    target: list[Tensor]

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Any | None = None,
        dist_sync_fn: Callable | None = None,
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
        self.register_buffer("fpr_limit", torch.tensor(fpr_limit))

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with new values.

        Args:
            preds (Tensor): predictions of the model
            target (Tensor): ground truth targets
        """
        self.target.append(target)
        self.preds.append(preds)

    def perform_cca(self) -> Tensor:
        """Perform the Connected Component Analysis on the self.target tensor.

        Raises:
            ValueError: ValueError is raised if self.target doesn't conform with requirements imposed by kornia for
                        connected component analysis.

        Returns:
            Tensor: Components labeled from 0 to N.
        """
        target = dim_zero_cat(self.target)

        # check and prepare target for labeling via kornia
        if target.min() < 0 or target.max() > 1:
            raise ValueError(
                f"kornia.contrib.connected_components expects input to lie in the interval [0, 1], but found "
                f"interval was [{target.min()}, {target.max()}]."
            )
        target = target.unsqueeze(1)  # kornia expects N1HW format
        target = target.type(torch.float)  # kornia expects FloatTensor
        if target.is_cuda:
            cca = connected_components_gpu(target)
        else:
            cca = connected_components_cpu(target)

        return cca

    def compute_pro(self, cca: Tensor, target: Tensor, preds: Tensor) -> tuple[Tensor, Tensor]:
        """Compute the pro/fpr value-pairs until the fpr specified by self.fpr_limit.

        It leverages the fact that the overlap corresponds to the tpr, and thus computes the overall
        PRO curve by aggregating per-region tpr/fpr values produced by ROC-construction.

        Returns:
            tuple[Tensor, Tensor]: tuple containing final fpr and tpr values.
        """

        # compute the global fpr-size
        fpr: Tensor = roc(preds, target)[0]  # only need fpr
        output_size = torch.where(fpr <= self.fpr_limit)[0].size(0)

        # compute the PRO curve by aggregating per-region tpr/fpr curves/values.
        tpr = torch.zeros(output_size, device=preds.device, dtype=torch.float)
        fpr = torch.zeros(output_size, device=preds.device, dtype=torch.float)
        new_idx = torch.arange(0, output_size, device=preds.device, dtype=torch.float)

        # Loop over the labels, computing per-region tpr/fpr curves, and aggregating them.
        # Note that, since the groundtruth is different for every all to `roc`, we also get
        # different/unique tpr/fpr curves (i.e. len(_fpr_idx) is different for every call).
        # We therefore need to resample per-region curves to a fixed sampling ratio (defined above).
        labels = cca.unique()[1:]  # 0 is background
        background = cca == 0
        _fpr: Tensor
        _tpr: Tensor
        for label in labels:
            interp: bool = False
            new_idx[-1] = output_size - 1
            mask = cca == label
            # Need to calculate label-wise roc on union of background & mask, as otherwise we wrongly consider other
            # label in labels as FPs. We also don't need to return the thresholds
            _fpr, _tpr = roc(preds[background | mask], mask[background | mask])[:-1]

            # catch edge-case where ROC only has fpr vals > self.fpr_limit
            if _fpr[_fpr <= self.fpr_limit].max() == 0:
                _fpr_limit = _fpr[_fpr > self.fpr_limit].min()
            else:
                _fpr_limit = self.fpr_limit

            _fpr_idx = torch.where(_fpr <= _fpr_limit)[0]
            # if computed roc curve is not specified sufficiently close to self.fpr_limit,
            # we include the closest higher tpr/fpr pair and linearly interpolate the tpr/fpr point at self.fpr_limit
            if not torch.allclose(_fpr[_fpr_idx].max(), self.fpr_limit):
                _tmp_idx = torch.searchsorted(_fpr, self.fpr_limit)
                _fpr_idx = torch.cat([_fpr_idx, _tmp_idx.unsqueeze_(0)])
                _slope = 1 - ((_fpr[_tmp_idx] - self.fpr_limit) / (_fpr[_tmp_idx] - _fpr[_tmp_idx - 1]))
                interp = True

            _fpr = _fpr[_fpr_idx]
            _tpr = _tpr[_fpr_idx]

            _fpr_idx = _fpr_idx.float()
            _fpr_idx /= _fpr_idx.max()
            _fpr_idx *= new_idx.max()

            if interp:
                # last point will be sampled at self.fpr_limit
                new_idx[-1] = _fpr_idx[-2] + ((_fpr_idx[-1] - _fpr_idx[-2]) * _slope)

            _tpr = self.interp1d(_fpr_idx, _tpr, new_idx)
            _fpr = self.interp1d(_fpr_idx, _fpr, new_idx)
            tpr += _tpr
            fpr += _fpr

        # Actually perform the averaging
        tpr /= labels.size(0)
        fpr /= labels.size(0)
        return fpr, tpr

    def _compute(self) -> tuple[Tensor, Tensor]:
        """Compute the PRO curve.

        Perform the Connected Component Analysis first then compute the PRO curve.

        Returns:
            tuple[Tensor, Tensor]: tuple containing final fpr and tpr values.
        """

        cca = self.perform_cca().flatten()
        target = dim_zero_cat(self.target).flatten()
        preds = dim_zero_cat(self.preds).flatten()

        return self.compute_pro(cca=cca, target=target, preds=preds)

    def compute(self) -> Tensor:
        """Fist compute PRO curve, then compute and scale area under the curve.

        Returns:
            Tensor: Value of the AUPRO metric
        """
        fpr, tpr = self._compute()

        aupro = auc(fpr, tpr, reorder=True)
        aupro = aupro / fpr[-1]  # normalize the area

        return aupro

    def generate_figure(self) -> tuple[Figure, str]:
        """Generate a figure containing the PRO curve and the AUPRO.

        Returns:
            tuple[Figure, str]: Tuple containing both the figure and the figure title to be used for logging
        """
        fpr, tpr = self._compute()
        aupro = self.compute()

        xlim = (0.0, self.fpr_limit.detach_().cpu().numpy())
        ylim = (0.0, 1.0)
        xlabel = "Global FPR"
        ylabel = "Averaged Per-Region TPR"
        loc = "lower right"
        title = "PRO"

        fig, _axis = plot_figure(fpr, tpr, aupro, xlim, ylim, xlabel, ylabel, loc, title)

        return fig, "PRO"

    @staticmethod
    def interp1d(old_x: Tensor, old_y: Tensor, new_x: Tensor) -> Tensor:
        """Function to interpolate a 1D signal linearly to new sampling points.

        Args:
            old_x (Tensor): original 1-D x values (same size as y)
            old_y (Tensor): original 1-D y values (same size as x)
            new_x (Tensor): x-values where y should be interpolated at

        Returns:
            Tensor: y-values at corresponding new_x values.
        """

        # Compute slope
        eps = torch.finfo(old_y.dtype).eps
        slope = (old_y[1:] - old_y[:-1]) / (eps + (old_x[1:] - old_x[:-1]))

        # Prepare idx for linear interpolation
        idx = torch.searchsorted(old_x, new_x)

        # searchsorted looks for the index where the values must be inserted
        # to preserve order, but we actually want the preceeding index.
        idx -= 1
        # we clamp the index, because the number of intervals = old_x.size(0) -1,
        # and the left neighbour should hence be at most number of intervals -1, i.e. old_x.size(0) - 2
        idx = torch.clamp(idx, 0, old_x.size(0) - 2)

        # perform actual linear interpolation
        y_new = old_y[idx] + slope[idx] * (new_x - old_x[idx])

        return y_new
