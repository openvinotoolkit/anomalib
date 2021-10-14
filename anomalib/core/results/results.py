"""
Result Set
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from torch import Tensor


@dataclass
class ClassificationResults:
    """
    Dataclass to store classification-task results.
    A classification task would return a anomaly
    classification score, which is used to compute
    the overall performance by comparing it with the
    true_labels (ground-truth).

    Args:
        filenames: List[Union[str, Path]]
        images: List[Union[np.ndarray, Tensor]]
        true_labels: List[Union[Tensor, np.ndarray]]
        anomaly_scores: List[Union[Tensor, np.ndarray]]
        performance: Dict[str, Any]

    Examples:
        >>> from anomalib.core.results import ClassificationResult
        >>> ClassificationResult()
        ClassificationResult(
            filenames=[], images=[],
            true_labels=[], anomaly_scores=[],
            performance={}
        )
    """

    filenames: List[Union[str, Path]] = field(default_factory=list)
    images: Optional[Tensor] = None
    true_labels: Optional[np.ndarray] = None
    pred_scores: Optional[np.ndarray] = None
    pred_labels: Optional[np.ndarray] = None
    # TODO: Use MetricCollection: https://jira.devtools.intel.com/browse/IAAALD-170
    performance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentationResults(ClassificationResults):
    """
    Dataclass to store segmentation-based task results.
    An anomaly segmentation task returns anomaly maps in
    addition to anomaly scores, which are then used to
    compute anomaly masks to compare against the true
    segmentation masks.

    Args:
        anomaly_maps: List[Union[np.ndarray, Tensor]]
        true_masks: List[Union[np.ndarray, Tensor]]
        pred_masks: List[Union[np.ndarray, Tensor]]

    Example:
        >>> from anomalib.core.results import SegmentationResult
        >>> SegmentationResult()
        SegmentationResult(
            true_labels=[], anomaly_scores=[], performance={},
            anomaly_maps=[], true_masks=[],
            pred_masks=[]
        )
    """

    anomaly_maps: Optional[np.ndarray] = None
    true_masks: Optional[np.ndarray] = None
    pred_masks: Optional[np.ndarray] = None
