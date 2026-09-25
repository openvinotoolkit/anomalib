# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch dataclasses for GRD-Net region-of-interest supervision."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import ClassVar, Generic, TypeVar

from torchvision.tv_tensors import Mask

from anomalib.data.dataclasses.generic import FieldDescriptor
from anomalib.data.dataclasses.numpy.image import NumpyImageBatch, NumpyImageItem
from anomalib.data.dataclasses.torch.image import ImageBatch, ImageItem
from anomalib.data.validators.torch.grdnet import GRDNetBatchValidator, GRDNetItemValidator

ROIMaskT = TypeVar("ROIMaskT", bound=Mask)
ROIPathT = TypeVar("ROIPathT", str, list[str])


@dataclass
class _ROIInputFields(Generic[ROIMaskT, ROIPathT], ABC):
    """Typed region-of-interest fields used by GRD-Net data containers."""

    roi_mask: FieldDescriptor[ROIMaskT | None] = FieldDescriptor(validator_name="validate_roi_mask")  # noqa: RUF009
    roi_mask_path: FieldDescriptor[ROIPathT | None] = FieldDescriptor(  # noqa: RUF009
        validator_name="validate_roi_mask_path",
    )

    @staticmethod
    @abstractmethod
    def validate_roi_mask(roi_mask: ROIMaskT | None) -> ROIMaskT | None:
        """Validate an ROI mask."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_roi_mask_path(roi_mask_path: ROIPathT | None) -> ROIPathT | None:
        """Validate an ROI mask path."""
        raise NotImplementedError


@dataclass
class GRDNetItem(
    GRDNetItemValidator,
    _ROIInputFields[Mask, str],
    ImageItem,
):
    """Image item with optional GRD-Net ROI supervision.

    The ROI is used only during training. Conversion to NumPy intentionally returns a standard
    :class:`~anomalib.data.dataclasses.numpy.image.NumpyImageItem` without ROI fields.
    """

    def __post_init__(self) -> None:
        """Validate that image and ROI spatial dimensions agree."""
        image = self.image
        roi_mask = self.roi_mask
        if image is None:
            msg = "GRD-Net items require an image."
            raise ValueError(msg)
        if roi_mask is not None and image.shape[-2:] != roi_mask.shape[-2:]:
            msg = f"Image and ROI spatial dimensions must match, got {tuple(image.shape)} and {tuple(roi_mask.shape)}."
            raise ValueError(msg)

    def to_numpy(self) -> NumpyImageItem:
        """Convert to a standard NumPy image item without training-only ROI fields.

        Returns:
            Standard NumPy image item containing the shared anomalib fields.
        """
        image_item = ImageItem(**{field.name: getattr(self, field.name) for field in fields(ImageItem)})
        return image_item.to_numpy()


@dataclass
class GRDNetBatch(
    GRDNetBatchValidator,
    _ROIInputFields[Mask, list[str]],
    ImageBatch,
):
    """Image batch with optional GRD-Net ROI supervision.

    The ROI masks use shape ``[B, H, W]``. Conversion to NumPy intentionally returns a standard
    :class:`~anomalib.data.dataclasses.numpy.image.NumpyImageBatch` without ROI fields.
    """

    item_class: ClassVar[type[GRDNetItem]] = GRDNetItem

    def __post_init__(self) -> None:
        """Validate that ROI fields match the image batch."""
        image = self.image
        roi_mask = self.roi_mask
        roi_mask_path = self.roi_mask_path
        if image is None:
            msg = "GRD-Net batches require an image batch."
            raise ValueError(msg)
        if roi_mask is not None and (image.shape[0] != roi_mask.shape[0] or image.shape[-2:] != roi_mask.shape[-2:]):
            msg = (
                "Image and ROI batch/spatial dimensions must match, "
                f"got {tuple(image.shape)} and {tuple(roi_mask.shape)}."
            )
            raise ValueError(msg)
        if roi_mask_path is not None and len(roi_mask_path) != image.shape[0]:
            msg = (
                f"Number of ROI mask paths ({len(roi_mask_path)}) does not match "
                f"the image batch size ({image.shape[0]})."
            )
            raise ValueError(msg)

    def to_numpy(self) -> NumpyImageBatch:
        """Convert to a standard NumPy image batch without training-only ROI fields.

        Returns:
            Standard NumPy image batch containing the shared anomalib fields.
        """
        image_batch = ImageBatch(**{field.name: getattr(self, field.name) for field in fields(ImageBatch)})
        return image_batch.to_numpy()
