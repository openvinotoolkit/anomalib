import torch

from dataclasses import dataclass, asdict
import warnings
from pathlib import Path

@dataclass
class BatchItem:
    image: torch.Tensor | None = None
    gt_label: int | None = None
    pred_label: int | None = None
    pred_score: float | None = None
    gt_mask: torch.Tensor | None = None
    pred_mask: torch.Tensor | None = None
    anomaly_map: torch.Tensor | None = None
    image_path: Path | None = None
    mask_path: Path | None = None
    pred_boxes: torch.Tensor | None = None
    box_scores: torch.Tensor | None = None
    box_labels: torch.Tensor | None = None
    video_path: Path | None = None
    original_image: torch.Tensor | None = None
    gt_boxes: torch.Tensor | None = None

    # def __getitem__(self, key: str) -> torch.Tensor:
    #     try:
    #         return asdict(self)[key]
    #     except KeyError:
    #         msg = f"{key} is not a valid key for StepOutput. Valid keys are: {list(asdict(self).keys())}"
    #         raise KeyError(msg)
        
    # def __setitem__(self, key, value):
    #     if hasattr(self, key):
    #         setattr(self, key, value)
    #     else:
    #         msg = f"{key} is not a valid key for StepOutput. Valid keys are: {list(asdict(self).keys())}"
    #         raise KeyError(msg)
    
    @property
    def mask(self) -> torch.Tensor:
        warnings.warn(
            "The `mask` attribute is deprecated and will be removed in the next release. "
            "Please use `pred_mask` instead.",
            DeprecationWarning,
        )
        return self.gt_mask
    
    @mask.setter
    def mask(self, value: torch.Tensor) -> None:
        warnings.warn(
            "The `mask` attribute is deprecated and will be removed in the next release. "
            "Please use `pred_mask` instead.",
            DeprecationWarning,
        )
        self.gt_mask = value