import torch

import numpy as np

from collections import namedtuple
from typing import NamedTuple, Generic, TypeVar

from dataclasses import dataclass, asdict
import warnings
from pathlib import Path


class InferenceBatch(NamedTuple):
    pred_score: torch.Tensor | None = None
    pred_label: torch.Tensor | None = None
    anomaly_map: torch.Tensor | None = None
    pred_mask: torch.Tensor | None = None


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

    def __post_init__(self):
        self.image = self.image
    
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


T = TypeVar("T")

@dataclass
class GenericInputBatch(Generic[T]):
    image: T | None = None

    gt_label: T | None = None
    gt_mask: T | None = None
    gt_boxes: T | None = None

    image_path: Path | None = None
    mask_path: Path | None = None
    video_path: Path | None = None
    original_image: T | None = None
    frames: T | None = None
    last_frame: int | None = None

@dataclass
class GenericOutputBatch(Generic[T]):
    pred_score: T | None = None
    pred_label: T | None = None
    anomaly_map: T | None = None
    pred_mask: T | None = None
    pred_boxes: T | None = None
    box_scores: T | None = None
    box_labels: T | None = None

@dataclass
class GenericBatch(Generic[T], GenericInputBatch[T], GenericOutputBatch[T]):
    pass

@dataclass
class NumpyBatch(GenericBatch[np.ndarray]):
        
    def __post_init__(self):
        self._format_and_validate()

    def _format_and_validate(self):

        # validate and format pred score
        if self.pred_score is not None:
            self.pred_score = np.squeeze(self.pred_score)

        # validate and format pred label
        if self.pred_label is not None:
            self.pred_label = np.squeeze(self.pred_label).astype(bool)

        # validate and format anomaly map
        if self.anomaly_map is not None:
            if self.anomaly_map.ndim == 4:
                assert self.anomaly_map.shape[1] == 1, f"Anomaly map must have 1 channel, got {self.anomaly_map.shape[1]}"
                self.anomaly_map = np.squeeze(self.anomaly_map, axis=1)
    


@dataclass
class TorchBatch(GenericBatch[torch.Tensor]):
    """Base class for storing the prediction results of a model."""

    def __post_init__(self):
        self._format_and_validate()

    def _format_and_validate(self):

        # validate and format pred score
        if self.pred_score is not None:
            self.pred_score = self.pred_score.squeeze()

        # validate and format pred label
        if self.pred_label is not None:
            self.pred_label = self.pred_label.squeeze().bool()

        # validate and format anomaly map
        if self.anomaly_map is not None:
            if self.anomaly_map.dim() == 4:
                assert self.anomaly_map.shape[1] == 1, f"Anomaly map must have 1 channel, got {self.anomaly_map.shape[1]}"
                self.anomaly_map = self.anomaly_map.squeeze(1)

@dataclass(kw_only=True)
class PredictBatch(InputBatch):
    """Base class for storing the prediction results of a model."""

    pred_score: torch.Tensor | None = None
    pred_label: torch.Tensor | None = None
    anomaly_map: torch.Tensor | None = None
    pred_mask: torch.Tensor | None = None
    pred_boxes: torch.Tensor | None = None
    box_scores: torch.Tensor | None = None
    box_labels: torch.Tensor | None = None

    def __post_init__(self):
        # compute pred score if not supplied
        if self.pred_score is None and self.anomaly_map is not None:
            # infer image scores from anomaly maps
            self.pred_score = torch.amax(self.anomaly_map, dim=(-2,-1)).squeeze()
        
        self._format_and_validate()

    def _format_and_validate(self):

        # validate and format pred score
        if self.pred_score is not None:
            self.pred_score = self.pred_score.squeeze()

        # validate and format pred label
        if self.pred_label is not None:
            self.pred_label = self.pred_label.squeeze().bool()

        # validate and format anomaly map
        if self.anomaly_map is not None:
            if self.anomaly_map.dim() == 4:  # anomaly map has shape [N, C, H, W]
                assert self.anomaly_map.shape[1] == 1, f"Anomaly map must have 1 channel, got {self.anomaly_map.shape[1]}"
                self.anomaly_map = self.anomaly_map.squeeze(1)
        
        # validate and format pred mask
        if self.pred_mask is not None:
            if self.pred_mask.dim() == 4:  # mask has shape [N, C, H, W]
                assert self.pred_mask.shape[1] == 1, f"Mask must have 1 channel, got {self.pred_mask.shape[1]}"
                self.pred_mask = self.pred_mask.squeeze(1)
            self.pred_mask = self.pred_mask.bool()

    def to_numpy():
        """Convert to NumpyPredictBatch"""
