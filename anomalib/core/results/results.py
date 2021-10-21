"""
Result Set
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
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
    true_labels: np.ndarray = np.empty(0)
    pred_scores: np.ndarray = np.empty(0)
    pred_labels: np.ndarray = np.empty(0)
    # TODO: Use MetricCollection: https://jira.devtools.intel.com/browse/IAAALD-170
    performance: Dict[str, Any] = field(default_factory=dict)

    def store_outputs(self, outputs: List[dict]):
        """
        Concatenate the outputs from the individual batches and store in the result set
        """
        if "image_path" in outputs[0].keys():
            self.filenames = [Path(f) for x in outputs for f in x["image_path"]]
        self.images = torch.vstack([x["image"] for x in outputs])
        self.true_labels = np.hstack([output["label"].cpu() for output in outputs])
        self.pred_scores = np.hstack([output["pred_scores"].cpu() for output in outputs])

    def evaluate(self, threshold: float):
        """
        Compute performance metrics
        """
        self.pred_labels = self.pred_scores >= threshold
        self.performance["image_f1_score"] = f1_score(self.true_labels, self.pred_labels)
        self.performance["balanced_accuracy_score"] = balanced_accuracy_score(self.true_labels, self.pred_labels)
        self.performance["image_roc_auc"] = roc_auc_score(self.true_labels, self.pred_scores)


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

    anomaly_maps: np.ndarray = np.empty(0)
    true_masks: np.ndarray = np.empty(0)
    pred_masks: Optional[np.ndarray] = None

    def store_outputs(self, outputs: List[dict]):
        """
        Concatenate the outputs from the individual batches and store in the result set
        """
        super().store_outputs(outputs)
        self.true_masks = np.vstack([output["mask"].squeeze(1).cpu() for output in outputs])
        self.anomaly_maps = np.vstack([output["anomaly_maps"].cpu() for output in outputs])

    def evaluate(self, threshold: float):
        """
        First compute common metrics, then compute segmentation-specific metrics
        """
        super().evaluate(threshold)
        self.performance["pixel_roc_auc"] = roc_auc_score(self.true_masks.flatten(), self.anomaly_maps.flatten())
