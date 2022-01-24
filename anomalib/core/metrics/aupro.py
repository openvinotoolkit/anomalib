"""Implementation of AUPRO score based on TorchMetrics."""
from typing import List, Optional, Callable, Any

import torch
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat
from skimage.measure import label, regionprops
from sklearn.metrics import auc
import numpy as np

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

class AUPRO(Metric):
    """Area under per region overlap (AUPRO) Metric."""
    is_differentiable = False
    pred: List[torch.Tensor]
    target: List[torch.Tensor]

    def __init__(
            self,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("pred", default=[], dist_reduce_fx="cat")  # pylint: disable=not-callable
        self.add_state("target", default=[], dist_reduce_fx="cat")  # pylint: disable=not-callable

    # pylint: disable=arguments-differ
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore
        """Update the state values."""
        self.pred.append(preds)
        self.target.append(target)

    def compute(self) -> torch.Tensor:
        """Compute the segmentation AUPRO Score.
        from https://github.com/YoungGod/DFR

        Returns:
            AUPRO Score
        """
        #print("###########")
        pred = dim_zero_cat(self.pred).numpy()
        target = dim_zero_cat(self.target).bool().numpy()

        with torch.no_grad():
            max_step = 1000
            expect_fpr = 0.3  # default 30%
            max_th = pred.max()
            min_th = pred.min()
            delta = (max_th - min_th) / max_step
            pros_mean = []
            threds = []
            fprs = []
            binary_score_maps = np.zeros_like(pred, dtype=np.bool)
            for step in range(max_step):
                thred = max_th - step * delta
                # segmentation
                binary_score_maps[pred <= thred] = 0
                binary_score_maps[pred > thred] = 1
                pro = []  # per region overlap
                # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
                for i in range(len(binary_score_maps)):  # for i th image
                    # pro (per region level)
                    label_map = label(target[i], connectivity=2)
                    props = regionprops(label_map)
                    for prop in props:
                        x_min, y_min, x_max, y_max = prop.bbox  # find the bounding box of an anomaly region
                        cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                        cropped_mask = prop.filled_image  # corrected!
                        intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                        pro.append(intersection / prop.area)
                pros_mean.append(np.array(pro).mean())
                # fpr for pro-auc
                targets_neg = np.invert(target)
                fpr = np.logical_and(targets_neg, binary_score_maps).sum() / targets_neg.sum()
                fprs.append(fpr)
                threds.append(thred)
            # as array
            pros_mean = np.array(pros_mean)
            fprs = np.array(fprs)
            # default 30% fpr vs pro, pro_auc
            idx = fprs <= expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
            fprs_selected = fprs[idx]
            fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
            pros_mean_selected = pros_mean[idx]
            seg_pro_auc = auc(fprs_selected, pros_mean_selected)

        return torch.tensor(seg_pro_auc)
