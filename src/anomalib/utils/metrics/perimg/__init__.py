"""Per-Image Metrics.

Overall approach:

    Thresholds are applied across all images, but each image is measured independently.
    In other words, thresholds are shared, but metrics are *per-image*.

    Thresholds are then indexed by a metric \\in [0, 1] so that any (model, dataset) can be compared.
    Key insight: the indexing metric is **only measured on normal images** in the test set.

        `PImO`: the shared metric is the mean of per-image FPR (`shared_fpr`).

    The indexing metric is then used as the X-axis of curve, where the Y-axis is the per-image metric.

        `PImO`: the Y-axis is the per-image TPR, or "Overlap" [between the predicted and ground-truth masks
        Therefore `PImO` stands for "Per-Image Overlap [curve]".

        Note: by definition, it is only defined on anomalous images.

    Finally, the area under each curve is computed.

        `PImO` --> `AUPImO` (Area Under the PImO curve).

    The shared metric is also used to restrict the threshold range.

        `PImO`: one can limit the upper bound (maximum value) of the shared FPR, which is the lower bound of thresholds.

    In such cases, the area under the curve is computed over the restricted range and normalized to [0, 1].
    Note: that this corresponds to taking the average value of the Y-axis over the restricted range.


Metrics here are generaly based on binary classification metrics (e.g. FPR, TPR, Precision) over a range of thresholds.

Several plot functions are provided to visualize these metrics.

Utilities are also provided to measure statistics over the per-image metric values, especially using boxplots.
"""

from .binclf_curve import PerImageBinClfCurve
from .common import compare_models_nonparametric, compare_models_parametric, perimg_boxplot_stats
from .pimo import AULogPImO, AUPImO, PImO

__all__ = [
    "PerImageBinClfCurve",
    "PImO",
    "AUPImO",
    "AULogPImO",
    "compare_models_nonparametric",
    "compare_models_parametric",
    "perimg_boxplot_stats",
]
