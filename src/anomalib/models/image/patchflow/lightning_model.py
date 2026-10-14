# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PatchFlow Lightning Model Implementation.

This model implements the PatchFlow algorithm for anomaly detection and localization.
PatchFlow models the distribution of patch-level feature embeddings using a learnable normalizing flow.

The model extracts patch embeddings from intermediate layers of a pretrained backbone.
It then passes them through a feature adaptor, which projects and aligns
features into a space suited for density estimation.
These patch features are then transformed by a flow-based density estimator that learns the distribution of normal data.

During inference, the model computes the likelihood of each patch under the learned flow.
Low-likelihood patches indicate anomalies.
The patch-level scores are aggregated to produce both an image-level anomaly score and a pixel-wise anomaly map.

Paper: https://arxiv.org/abs/2602.05238

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Patchflow
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = Patchflow()
    >>> engine = Engine()

    >>> engine.fit(model, datamodule=datamodule)  # doctest: +SKIP
    >>> predictions = engine.predict(model, datamodule=datamodule)  # doctest: +SKIP
"""

from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim
from torchvision.transforms.v2 import Compose, Normalize, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import AUROC, Evaluator, F1Score
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .loss import PatchflowLoss
from .torch_model import PatchflowModel


class Patchflow(AnomalibModule):
    """Lightning Module for the PatchFlow algorithm.

    Args:
        backbone: Backbone network. A timm model name or a DINOv2 model name.
            Defaults to ``"tf_efficientnet_b5"``.
        pre_trained: Whether to use pre-trained backbone weights.
            Defaults to ``True``.
        flow_steps: Number of coupling blocks in the normalizing flow.
            Defaults to ``1``.
        flow_feature_dim: Channel dimension after the feature adaptor.
            Defaults to ``128``.
        num_scales: Number of input resolutions for multi-scale extraction.
            Defaults to ``3``.
        patch_size: Kernel size of the AvgPool for local aggregation.
            Defaults to ``3``.
        flow_hidden_dim: Hidden channels in the flow subnet.
            Defaults to ``128``.
        crop_size: Optional center crop size ``(H, W)``. When set, the
            input is center-cropped before feature extraction and the
            anomaly map is padded with value ``-1`` back to the original input size.
            Defaults to ``None``.
        lr: Learning rate for the Adam optimizer.
            Defaults to ``0.001``.
        weight_decay: Weight decay (L2 penalty) for the Adam optimizer.
            Defaults to ``0.0001``.
        pre_processor: Pre-processor instance or boolean.
            Defaults to ``True``.
        post_processor: Post-processor instance or boolean.
            Defaults to ``True``.
        evaluator: Evaluator instance or boolean.
            Defaults to ``True``.
        visualizer: Visualizer instance or boolean.
            Defaults to ``True``.
    """

    def __init__(
        self,
        backbone: str = "tf_efficientnet_b5",  # supported: "tf_efficientnet_b5", "vit_base_patch14_dinov2"
        pre_trained: bool = True,
        flow_steps: int = 1,
        flow_feature_dim: int = 128,
        num_scales: int = 3,
        patch_size: int = 3,
        flow_hidden_dim: int = 128,
        crop_size: tuple[int, int] | None = None,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
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
        if self.input_size is None:
            msg = "Patchflow needs input size to build torch model."
            raise ValueError(msg)

        self.model = PatchflowModel(
            input_size=self.input_size,
            backbone=backbone,
            pre_trained=pre_trained,
            flow_steps=flow_steps,
            flow_feature_dim=flow_feature_dim,
            num_scales=num_scales,
            patch_size=patch_size,
            flow_hidden_dim=flow_hidden_dim,
            crop_size=crop_size,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss = PatchflowLoss()

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the training step.

        Args:
            batch: Input batch.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            Dictionary containing the loss value.
        """
        del args, kwargs

        hidden_variables, jacobians = self.model(batch.image)
        loss = self.loss(hidden_variables, jacobians)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step.

        Args:
            batch: Input batch.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            Batch dictionary containing anomaly maps.
        """
        del args, kwargs

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return PatchFlow trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer.

        Returns:
            Adam optimizer.
        """
        return optim.Adam(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    @staticmethod
    def configure_pre_processor(image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure the default pre-processor for PatchFlow.

        Matches the official repo transforms (Resize + ImageNet Normalize).
        https://github.com/BOX-LEO/PatchFlow/blob/main/data.py

        Note:
            The default ``(768, 768)`` is not divisible by the DINOv2 ViT
            patch size (14).  When a DINOv2 backbone is used, the torch model
            automatically center-crops the input to the nearest
            patch-aligned size, so no manual ``crop_size`` is required.

        Args:
            image_size: Target image size. Defaults to ``(768, 768)``
                per the paper (Section 4.3).

        Returns:
            PreProcessor with the configured transforms.
        """
        image_size = image_size or (768, 768)
        return PreProcessor(
            transform=Compose([
                Resize(image_size, antialias=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        )

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: One-class learning.
        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_evaluator() -> Evaluator:
        """Configure default evaluator with image/pixel AUROC and F1."""
        image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
        val_metrics = [image_auroc, pixel_auroc]

        image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
        pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
        pixel_f1score = F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_")
        test_metrics = [image_auroc, image_f1score, pixel_auroc, pixel_f1score]
        return Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)
