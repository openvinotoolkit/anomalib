"""Per-Image Metrics.

Overall approach:
Thresholds are computed across all images, but the metrics are computed per-image.
Metrics here are based on binary classification metrics (e.g. FPR, TPR, Precision) over a range of thresholds.
"""

from .binclf_curve import PerImageBinClfCurve
from .pimo import AUPImO, PImO

# TODO import in the parent module
__all__ = [
    "PerImageBinClfCurve",
    "PImO",
    "AUPImO",
]
