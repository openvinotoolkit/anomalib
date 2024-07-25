import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generic, NamedTuple, TypeVar

import numpy as np
import torch


class InferenceBatch(NamedTuple):
    pred_score: torch.Tensor | None = None
    pred_label: torch.Tensor | None = None
    anomaly_map: torch.Tensor | None = None
    pred_mask: torch.Tensor | None = None


class BackwardCompatibilityMixin:
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

    @property
    def mask(self) -> torch.Tensor:
        """Legacy getter for gt_mask. Will be removed in v1.2."""
        warnings.warn(
            "The `mask` attribute is deprecated and will be removed in v1.2. " "Please use `pred_mask` instead.",
            DeprecationWarning,
        )
        return self.gt_mask

    @mask.setter
    def mask(self, value: torch.Tensor) -> None:
        """Legacy setter for gt_mask. Will be removed in v1.2."""
        warnings.warn(
            "The `mask` attribute is deprecated and will be removed in v1.2. " "Please use `pred_mask` instead.",
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


@dataclass(kw_only=True)
class NumpyBatch(GenericBatch[np.ndarray]):
    def __post_init__(self):
        self._format_and_validate()

    def _format_and_validate(self):
        # validate and format image
        assert self.image.ndim == 4, "Image must have shape [N, H, W, C]"
        if self.image.shape[1] == 3:
            self.image = self.image.transpose(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]

        # validate and format pred score
        if self.pred_score is not None:
            self.pred_score = np.squeeze(self.pred_score)

        # validate and format pred label
        if self.pred_label is not None:
            self.pred_label = np.squeeze(self.pred_label).astype(bool)

        # validate and format anomaly map
        if self.anomaly_map is not None:
            if self.anomaly_map.ndim == 4:
                assert (
                    self.anomaly_map.shape[1] == 1
                ), f"Anomaly map must have 1 channel, got {self.anomaly_map.shape[1]}"
                self.anomaly_map = np.squeeze(self.anomaly_map, axis=1)


@dataclass(kw_only=True)
class Batch(BackwardCompatibilityMixin, GenericBatch[torch.Tensor]):
    """Base class for storing the prediction results of a model."""

    def __post_init__(self):
        # compute pred score if not supplied
        if self.pred_score is None and self.anomaly_map is not None:
            # infer image scores from anomaly maps
            self.pred_score = torch.amax(self.anomaly_map, dim=(-2, -1)).squeeze()
        self._format_and_validate()

    def _format_and_validate(self):
        # validate and image
        assert self.image.dim() == 4, "Image must have shape [N, C, H, W]"

        # validate and format pred score
        if self.pred_score is not None:
            self.pred_score = self.pred_score.squeeze()

        # validate and format pred label
        if self.pred_label is not None:
            self.pred_label = self.pred_label.squeeze().bool()

        # validate and format anomaly map
        if self.anomaly_map is not None:
            if self.anomaly_map.dim() == 4:  # anomaly map has shape [N, C, H, W]
                assert (
                    self.anomaly_map.shape[1] == 1
                ), f"Anomaly map must have 1 channel, got {self.anomaly_map.shape[1]}"
                self.anomaly_map = self.anomaly_map.squeeze(1)

        # validate and format pred mask
        if self.pred_mask is not None:
            if self.pred_mask.dim() == 4:  # mask has shape [N, C, H, W]
                assert self.pred_mask.shape[1] == 1, f"Mask must have 1 channel, got {self.pred_mask.shape[1]}"
                self.pred_mask = self.pred_mask.squeeze(1)
            self.pred_mask = self.pred_mask.bool()

    def to_numpy(self) -> NumpyBatch:
        """Convert the batch to a NumpyBatch object."""
        batch_dict = asdict(self)
        for key, value in batch_dict.items():
            if isinstance(value, torch.Tensor):
                batch_dict[key] = value.cpu().numpy()
        return NumpyBatch(
            **batch_dict,
        )
