import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, astuple, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Generic, NamedTuple, TypeVar

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


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
            "The `mask` attribute is deprecated and will be removed in v1.2. Please use `pred_mask` instead.",
            DeprecationWarning,
        )
        return self.gt_mask

    @mask.setter
    def mask(self, value: torch.Tensor) -> None:
        """Legacy setter for gt_mask. Will be removed in v1.2."""
        warnings.warn(
            "The `mask` attribute is deprecated and will be removed in v1.2. Please use `pred_mask` instead.",
            DeprecationWarning,
        )
        self.gt_mask = value


class ReplaceMixin:  # InPlaceUpdateMixin?
    """Mixin class for dataclasses that allows for in-place replacement of attributes."""

    def replace(self, in_place=True, **changes):  # update?
        """Replace fields in place and call __post_init__ to reinitialize the instance.

        Parameters:
        changes (dict): A dictionary of field names and their new values.
        """
        if not is_dataclass(self):
            raise TypeError("replace can only be used with dataclass instances")

        if in_place:
            for field in fields(self):
                if field.init and field.name in changes:
                    setattr(self, field.name, changes[field.name])
            if hasattr(self, "__post_init__"):
                self.__post_init__()
            return self
        else:
            return replace(self, **changes)


T = TypeVar("T", torch.Tensor, np.ndarray)


@dataclass
class GenericInput(Generic[T]):
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
class GenericOutput(Generic[T]):
    pred_score: T | None = None
    pred_label: T | None = None
    anomaly_map: T | None = None
    pred_mask: T | None = None
    pred_boxes: T | None = None
    box_scores: T | None = None
    box_labels: T | None = None


@dataclass
class GenericDatasetItem(Generic[T], GenericInput[T], GenericOutput[T]):
    def __post_init__(self):
        if self.image is not None:
            assert self.image.ndim == 3, "Image must have shape [C, H, W] or [H, W, C]"

        if self.anomaly_map is not None:
            assert (
                self.anomaly_map.ndim == 2 or self.anomaly_map.ndim == 3
            ), "Anomaly map must have shape [H, W] or [C, H, W]"
            if self.anomaly_map.ndim == 3:
                assert (
                    self.anomaly_map.shape[0] == 1 or self.anomaly_map.shape[-1] == 1
                ), f"Anomaly map must have 1 channel, got shape {self.anomaly_map.shape}"
                self.anomaly_map = self.anomaly_map.squeeze(1)


@dataclass
class NumpyDatasetItem(
    ReplaceMixin,
    GenericDatasetItem[np.ndarray],
):
    def __post_init__(self):
        GenericDatasetItem.__post_init__(self)

        # validate and format image
        assert self.image.ndim == 3, "Image must have shape [H, W, C]"
        if self.image.shape[0] == 3:
            self.image = self.image.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]


@dataclass
class DatasetItem(
    BackwardCompatibilityMixin,
    ReplaceMixin,
    GenericDatasetItem[torch.Tensor],
):
    """Base class for storing"""

    def __post_init__(self):
        GenericDatasetItem.__post_init__(self)

        # validate and format image
        if self.image.shape[2] == 3:
            self.image = self.image.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

    def to_numpy(self) -> NumpyDatasetItem:
        """Convert the dataset item to a NumpyBatch object."""
        batch_dict = asdict(self)
        for key, value in batch_dict.items():
            if isinstance(value, torch.Tensor):
                batch_dict[key] = value.cpu().numpy()
        return NumpyDatasetItem(
            **batch_dict,
        )


@dataclass
class GenericBatch(Generic[T], GenericInput[T], GenericOutput[T], ABC):
    def __post_init__(self):
        if self.image is not None:
            assert self.image.ndim == 4, f"Image must have shape [N, C, H, W] or [N, H, W, C], got {self.image.shape}"

        if self.anomaly_map is not None:
            if self.anomaly_map.ndim == 4:
                assert (
                    self.anomaly_map.shape[1] == 1
                ), f"Anomaly map must have 1 channel, got {self.anomaly_map.shape[1]}"
                self.anomaly_map = self.anomaly_map.squeeze(1)

        if self.pred_score is not None:
            if self.pred_score.ndim == 2:
                self.pred_score = self.pred_score.squeeze(0)
            elif self.pred_score.ndim == 0:
                self.pred_score = self.pred_score.unsqueeze(0)
            elif self.pred_score.ndim != 1:
                raise ValueError(f"Invalid shape for pred_score: {self.pred_score.shape}")

        if self.pred_label is not None:
            if self.pred_label.ndim == 2:
                self.pred_label = self.pred_label.squeeze(0)
            elif self.pred_label.ndim == 0:
                self.pred_label = self.pred_label.unsqueeze(0)
            elif self.pred_label.ndim != 1:
                raise ValueError(f"Invalid shape for pred_label: {self.pred_label.shape}")

    @property
    def batch_size(self) -> int | None:
        for item in astuple(self):
            if hasattr(item, "shape"):
                return item.shape[0]
            elif hasattr(item, "__len__"):
                return len(item)
        return None

    @property
    @abstractmethod
    def items(self):
        pass

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        yield from self.items


@dataclass(kw_only=True)
class NumpyBatch(
    ReplaceMixin,
    GenericBatch[np.ndarray],
):
    def __post_init__(self):
        GenericBatch.__post_init__(self)

        # validate and format image
        assert self.image.ndim == 4, "Image must have shape [N, H, W, C]"
        if self.image.shape[1] == 3:
            self.image = self.image.transpose(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]

        # validate and format pred label
        if self.pred_label is not None:
            self.pred_label = self.pred_label.astype(bool)

        # validate and format anomaly map
        if self.anomaly_map is not None:
            if self.anomaly_map.ndim == 4:
                assert (
                    self.anomaly_map.shape[1] == 1
                ), f"Anomaly map must have 1 channel, got {self.anomaly_map.shape[1]}"
                self.anomaly_map = np.squeeze(self.anomaly_map, axis=1)

    @property
    def items(self):
        """Convert the batch to a list of DatasetItem objects."""
        batch_dict = asdict(self)
        items = []
        for i in range(self.batch_size):
            items.append(
                NumpyDatasetItem(
                    **{key: value[i] if isinstance(value, Iterable) else None for key, value in batch_dict.items()},
                ),
            )
        return items


@dataclass(kw_only=True)
class Batch(
    BackwardCompatibilityMixin,
    ReplaceMixin,
    GenericBatch[torch.Tensor],
):
    """Base class for storing the prediction results of a model."""

    def __post_init__(self):
        GenericBatch.__post_init__(self)

        # validate and format image
        assert self.image.dim() in [3, 4], "Image must have shape [N, C, H, W] or [C, H, W]"
        if self.image.dim() == 3:
            self.image = self.image.unsqueeze(0)

        # validate and format pred score
        if self.pred_score is None and self.anomaly_map is not None:
            # infer image scores from anomaly maps
            self.pred_score = torch.amax(self.anomaly_map, dim=(-2, -1))

        # validate and format pred label
        if self.pred_label is not None:
            self.pred_label = self.pred_label.bool()

        # validate and format anomaly map
        if self.anomaly_map is not None:
            assert self.anomaly_map.ndim in [
                2,
                3,
                4,
            ], f"Anomaly map must have shape [N, 1, H, W], [N, H, W] or [H, W], got {self.anomaly_map.shape}"
            if self.anomaly_map.ndim == 4:  # anomaly map has shape [N, C, H, W]
                assert (
                    self.anomaly_map.shape[1] == 1
                ), f"Anomaly map must have 1 channel, got {self.anomaly_map.shape[1]}"
                self.anomaly_map = self.anomaly_map.squeeze(1)
            elif self.anomaly_map.ndim == 2:
                self.anomaly_map = self.anomaly_map.unsqueeze(0)

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

    @property
    def items(self):
        """Convert the batch to a list of DatasetItem objects."""
        batch_dict = asdict(self)
        items = []
        for i in range(self.batch_size):
            items.append(
                DatasetItem(
                    **{key: value[i] if isinstance(value, Iterable) else None for key, value in batch_dict.items()},
                ),
            )
        return items

    @classmethod
    def collate(cls, items: list[DatasetItem]):
        """Convert a list of DatasetItem objects to a Batch object."""
        keys = [key for key, value in asdict(items[0]).items() if value is not None]
        out_dict = {key: default_collate([getattr(item, key) for item in items]) for key in keys}
        return cls(**out_dict)
