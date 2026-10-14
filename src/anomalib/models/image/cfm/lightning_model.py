# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning wrapper for CFM (Cross-modal Feature Mapping)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.nn import functional
from torchvision.transforms.v2 import Resize

from anomalib import LearningType
from anomalib.data import Batch, InferenceBatch
from anomalib.models import AnomalibModule
from anomalib.pre_processing import PreProcessor

from .torch_model import CFMModel

if TYPE_CHECKING:
    from pathlib import Path

    from lightning.pytorch.utilities.types import STEP_OUTPUT

    from anomalib.metrics import Evaluator
    from anomalib.post_processing import PostProcessor
    from anomalib.visualization import Visualizer


class CFM(AnomalibModule):
    """AnomalibModule wrapper for CFM model.

    Args:
        lr: Learning rate.
        rgb_backbone: Name of the backbone DINO for RGB.
        group_size: Dimension of the groups for PointTransformer.
        num_group: Number of groups for PointTransformer.
        pointmae_weights: Path to Point-MAE pretrained weights. If ``None``,
            weights are automatically downloaded to the anomalib cache.
        pre_processor: Pre-processor used to transform input data before
            passing to model.
        post_processor: Post-processor used to process model predictions.
        evaluator: Evaluator used to compute metrics.
        visualizer: Visualizer used to create visualizations.

    Note:
        Use a depth datamodule such as ``MVTec3D`` and set ``category`` to train on a
        single object class. See ``examples/configs/model/cfm.yaml``.

    Example:
        >>> from anomalib.models import CFM
        >>> model = CFM(lr=1e-4, num_group=512)
    """

    def __init__(
        self,
        lr: float = 1e-4,
        rgb_backbone: str = "vit_base_patch8_224.dino",
        group_size: int = 128,
        num_group: int = 1024,
        pointmae_weights: str | Path | None = None,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.save_hyperparameters(ignore=["pre_processor", "post_processor", "evaluator", "visualizer"])
        self.lr = lr

        # Initialization of core model
        self.model: CFMModel = CFMModel(
            rgb_backbone=rgb_backbone,
            group_size=group_size,
            num_group=num_group,
            pointmae_weights=pointmae_weights,
        )

    @staticmethod
    def configure_pre_processor(image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure the pre-processor dynamically based on config/data."""
        size = image_size if image_size is not None else (224, 224)
        return PreProcessor(transform=Resize(size))

    @property
    def learning_type(self) -> LearningType:
        """Returns the model's learning type (One-Class)."""
        return LearningType.ONE_CLASS

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Returns specific arguments for the trainer."""
        return {}

    @staticmethod
    def _get_first_tensor(batch: Batch, keys: tuple[str, ...]) -> torch.Tensor | None:
        """Return the first non-None tensor found for the given keys.

        Prefers public interfaces:
        - Mapping access (``batch[key]`` / ``batch.get(key)``)
        - Attribute access (``batch.key``)

        Falls back to item access for non-mapping, indexable objects to stay compatible
        with custom batch types.
        """
        if isinstance(batch, Mapping):
            for key in keys:
                value = cast("Mapping[str, Any]", batch).get(key)
                if value is not None:
                    return cast("torch.Tensor", value)
            return None

        for key in keys:
            value = getattr(batch, key, None)
            if value is not None:
                return cast("torch.Tensor", value)

        for key in keys:
            try:
                value = cast("Any", batch)[key]
            except (TypeError, KeyError):
                continue
            if value is not None:
                return cast("torch.Tensor", value)

        return None

    @classmethod
    def _get_data(cls, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract both the RGB image and the Point Cloud from multimodal batch.

        Supports both dict-style batches with 'point_cloud' key and anomalib
        DepthBatch dataclasses where 3D data is stored under 'depth_map'.
        """
        rgb = cls._get_first_tensor(batch, ("image",))
        xyz = cls._get_first_tensor(batch, ("point_cloud", "depth_map"))
        if rgb is not None and xyz is not None:
            if xyz.ndim != 4 or xyz.shape[1] != 3:
                msg = f"CFM requires 3D input with shape (B, 3, H, W), got {xyz.shape}"
                raise TypeError(msg)
            if xyz.shape[-2:] != rgb.shape[-2:]:
                xyz = functional.interpolate(xyz, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
            return rgb, xyz

        msg = "Tensor 'image' and 'point_cloud'/'depth_map' not found in the batch."
        raise KeyError(msg)

    def forward(self, batch: torch.Tensor | Batch, *_args: object, **_kwargs: object) -> InferenceBatch:
        """Forward pass used by predict/export code paths.

        Notes:
            `AnomalibModule`'s default `forward` assumes single-input models and calls
            `self.model(image)`. CFM is multimodal, so we override `forward` to require
            a `Batch` or mapping with both 'image' and 'point_cloud'/'depth_map' keys.

        Raises:
            TypeError: If called with a raw image tensor (missing 3D modality).
            TypeError: If the underlying model did not return an ``InferenceBatch``
                (e.g. model is in training mode).
        """
        if isinstance(batch, torch.Tensor):
            msg = (
                "CFM requires both RGB and 3D point cloud inputs. "
                "Pass a Batch/dict with 'image' and 'point_cloud' keys."
            )
            raise TypeError(msg)
        rgb, xyz = self._get_data(batch)

        out = self.model(rgb, xyz)
        if isinstance(out, InferenceBatch):
            return out
        msg = "CFM forward is expected to run in eval mode and return an InferenceBatch."
        raise TypeError(msg)

    def training_step(self, batch: Batch, _batch_idx: int, *_args: object, **_kwargs: object) -> torch.Tensor:
        """Executes a training step evaluating the loss between multimodal projections."""
        rgb, xyz = self._get_data(batch)
        out = self.model(rgb, xyz)

        loss = out["loss"]
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=rgb.shape[0])
        return loss

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Run validation / test / predict: forward plus attach scores and anomaly maps to the batch."""
        del args, kwargs

        rgb, xyz = self._get_data(batch)
        out = self.model(rgb, xyz)

        return batch.update(pred_score=out.pred_score, anomaly_map=out.anomaly_map)

    def test_step(self, batch: Batch, batch_idx: int, *args: object, **kwargs: object) -> STEP_OUTPUT:
        """Same as ``validation_step`` (multimodal; do not use the image-only base implementation)."""
        return self.validation_step(batch, batch_idx, *args, **kwargs)

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT:
        """Same as ``validation_step`` (multimodal; do not use the image-only base implementation)."""
        del dataloader_idx
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configuration of the optimizer (Adam) for trainable modules."""
        # Optimization of mapper parameters only (projection nets)
        return torch.optim.Adam(
            params=cast("CFMModel", self.model).mapper_parameters(),
            lr=self.lr,
        )
