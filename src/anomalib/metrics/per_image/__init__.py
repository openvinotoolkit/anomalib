"""Per-Image Metrics.

TODO(jpcbertoldo): add formalities (license header, author)
"""

from .binclf_curve import per_image_binclf_curve, per_image_fpr, per_image_tpr
from .binclf_curve_numpy import BinclfAlgorithm, BinclfThreshsChoice
from .pimo import AUPIMO, PIMO, AUPIMOResult, PIMOResult, aupimo_scores, pimo_curves
from .pimo_numpy import PIMOSharedFPRMetric
from .utils import per_image_scores_stats
from .utils_numpy import StatsOutliersPolicy, StatsRepeatedPolicy

__all__ = [
    # constants
    "BinclfAlgorithm",
    "BinclfThreshsChoice",
    "StatsOutliersPolicy",
    "StatsRepeatedPolicy",
    "PIMOSharedFPRMetric",
    # result classes
    "PIMOResult",
    "AUPIMOResult",
    # functional interfaces
    "per_image_binclf_curve",
    "per_image_fpr",
    "per_image_tpr",
    "pimo_curves",
    "aupimo_scores",
    "per_image_scores_stats",
    # torchmetrics interfaces
    "PIMO",
    "AUPIMO",
]
