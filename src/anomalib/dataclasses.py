import torch

from collections import namedtuple

from dataclasses import dataclass, asdict, InitVar
import warnings
from pathlib import Path

from lightning.pytorch.trainer.states import RunningStage

# from anomalib.models.components.base import AnomalyModule


InferenceBatch = namedtuple(
    "InferenceBatch",
    [
        "pred_label",
        "pred_score",
        "pred_mask",
        "anomaly_map"
    ]
)


@dataclass
class BatchItem:
    """Base class for dataclass objects that are passed between steps in the pipeline."""

    def __getitem__(self, key: str) -> torch.Tensor:
        try:
            return asdict(self)[key]
        except KeyError:
            msg = f"{key} is not a valid key for StepOutput. Valid keys are: {list(asdict(self).keys())}"
            raise KeyError(msg)
        
    def __setitem__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            msg = f"{key} is not a valid key for StepOutput. Valid keys are: {list(asdict(self).keys())}"
            raise KeyError(msg)


@dataclass(kw_only=True)
class InputBatch(BatchItem):
    """Base class for storing the input data of a model."""

    image: torch.Tensor | None = None

    gt_label: int | None = None
    gt_mask: torch.Tensor | None = None
    gt_boxes: torch.Tensor | None = None

    image_path: Path | None = None
    mask_path: Path | None = None
    video_path: Path | None = None
    original_image: torch.Tensor | None = None
    frames: torch.Tensor | None = None
    last_frame: int | None = None
    
    @property
    def mask(self) -> torch.Tensor:
        """Legacy getter for gt_mask. Will be removed in v1.2."""
        warnings.warn(
            "The `mask` attribute is deprecated and will be removed in v1.2. "
            "Please use `pred_mask` instead.",
            DeprecationWarning,
        )
        return self.gt_mask
    
    @mask.setter
    def mask(self, value: torch.Tensor) -> None:
        """Legacy setter for gt_mask. Will be removed in v1.2."""
        warnings.warn(
            "The `mask` attribute is deprecated and will be removed in v1.2. "
            "Please use `pred_mask` instead.",
            DeprecationWarning,
        )
        self.gt_mask = value


@dataclass(kw_only=True)
class PredictBatch(InputBatch):
    """Base class for storing the prediction results of a model."""

    pred_score: float | None = None
    pred_label: int | None = None
    anomaly_map: torch.Tensor | None = None
    pred_mask: torch.Tensor | None = None
    pred_boxes: torch.Tensor | None = None
    box_scores: torch.Tensor | None = None
    box_labels: torch.Tensor | None = None

    def __post_init__(self):
        # compute pred score if not supplied
        if self.pred_score is None and self.anomaly_map is not None:
            # infer image scores from anomaly maps
            self.pred_score = (
                self.anomaly_map  # noqa: PD011
                .reshape(self.anomaly_map.shape[0], -1)
                .max(dim=1)
                .values
            )
