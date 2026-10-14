# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""FoundAD: Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors.

This module implements the FoundAD model for anomaly detection. The method trains
a lightweight manifold projector on top of a frozen DINOv2 encoder, using CutPaste
augmentation to provide off-manifold supervision. Anomaly detection is performed
by measuring the MSE between encoder features and projector output — high MSE
indicates that features lie off the normal image manifold.

The model is particularly effective in few-shot settings where only a small
number of normal training samples are available.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import FoundAD
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = FoundAD()
    >>> engine = Engine()

    >>> engine.fit(model, datamodule=datamodule)  # doctest: +SKIP
    >>> predictions = engine.predict(model, datamodule=datamodule)  # doctest: +SKIP

Reference:
    Zhai et al., "Foundation Visual Encoders Are Secretly Few-Shot Anomaly
    Detectors", ICLR 2026. https://arxiv.org/abs/2510.01934

See Also:
    :class:`anomalib.models.image.foundad.torch_model.FoundADModel`:
        PyTorch implementation of the FoundAD model.
"""

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.models.image.foundad.components.augmentation import FewShotAugmentation
from anomalib.models.image.foundad.components.cutpaste import CutPasteUnion
from anomalib.models.image.foundad.torch_model import FoundADModel
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

logger = logging.getLogger(__name__)

# Default image size for DINOv2 with patch_size=14.
# 518 = 37 * 14, giving 37x37 = 1369 patches.
DEFAULT_IMAGE_SIZE = 518

# Default image size for DINOv3 with patch_size=16 (518 is not divisible by 16).
# 512 = 32 * 16, giving 32x32 = 1024 patches.
DINOV3_DEFAULT_IMAGE_SIZE = 512

# Training defaults
TRAINING_DEFAULTS: dict[str, Any] = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "color_jitter": 0.5,
}


class FoundAD(AnomalibModule):
    """FoundAD Lightning Module.

    Trains a lightweight manifold projector on a frozen DINOv2 encoder with
    CutPaste augmentation. Only the projector parameters are learned.

    Args:
        encoder_name: DINOv2 model name. Defaults to ``"dinov2_vit_base_14"``.
        pred_depth: Number of projector transformer blocks. Defaults to ``6``.
        pred_emb_dim: Projector internal dimension. Defaults to ``384``.
        n_layer: Encoder layer to extract features from (n-th from last).
            Defaults to ``3``.
        top_k: Top-K patches for image-level scoring. Defaults to ``10``.
        dropout: Dropout on features before projector. Defaults to ``0.2``.
        feat_normed: L2-normalize features. Defaults to ``False``.
        use_pos_embed: Use positional encoding in projector. Defaults to ``False``.
        image_size: Input image size. Defaults to ``518`` for DINOv2 (37*14,
            patch size 14) or ``512`` for DINOv3 (32*16, patch size 16) encoders.
            Smaller values like ``224`` speed up inference at cost of resolution,
            but must stay divisible by the encoder's patch size.
        lr: Learning rate. Defaults to ``1e-3``.
        weight_decay: Weight decay for AdamW. Defaults to ``1e-4``.
        color_jitter: CutPaste color jitter strength. Defaults to ``0.5``.
        use_few_shot_augmentation: Apply random orientation (flip/rotate) and
            appearance (color jitter/grayscale/blur) augmentation to the base
            training images before CutPaste. This matches the paper's default
            training protocol and is important for expanding the diversity of
            a tiny (1/2/4-shot) training set; disabling it will hurt few-shot
            results. Defaults to ``True``.
        pre_processor: Pre-processor configuration.
        post_processor: Post-processor configuration.
        evaluator: Evaluator configuration.
        visualizer: Visualizer configuration.

    Example:
        >>> from anomalib.data import MVTecAD
        >>> from anomalib.models import FoundAD
        >>> from anomalib.engine import Engine

        >>> model = FoundAD()
        >>> datamodule = MVTecAD()
        >>> engine = Engine()
        >>> engine.fit(model, datamodule=datamodule)  # doctest: +SKIP
    """

    def __init__(
        self,
        encoder_name: str = "dinov2_vit_base_14",
        pred_depth: int = 6,
        pred_emb_dim: int = 384,
        n_layer: int = 3,
        top_k: int = 10,
        dropout: float = 0.2,
        feat_normed: bool = False,
        use_pos_embed: bool = False,
        image_size: int | None = None,
        lr: float = TRAINING_DEFAULTS["lr"],
        weight_decay: float = TRAINING_DEFAULTS["weight_decay"],
        color_jitter: float = TRAINING_DEFAULTS["color_jitter"],
        use_few_shot_augmentation: bool = True,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        # Resolve the default image size from the encoder family (DINOv3 uses
        # patch size 16, so it needs a different default than DINOv2's 518).
        if image_size is None:
            image_size = DINOV3_DEFAULT_IMAGE_SIZE if encoder_name.startswith("dinov3") else DEFAULT_IMAGE_SIZE

        if pre_processor is True:
            pre_processor = FoundAD.configure_pre_processor(image_size=(image_size, image_size))

        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.model = FoundADModel(
            encoder_name=encoder_name,
            pred_depth=pred_depth,
            pred_emb_dim=pred_emb_dim,
            n_layer=n_layer,
            top_k=top_k,
            dropout=dropout,
            feat_normed=feat_normed,
            use_pos_embed=use_pos_embed,
            image_size=image_size,
        )

        self._image_size = image_size
        self.lr = lr
        self.weight_decay = weight_decay

        # CutPaste augmentation (training only, not part of model state)
        self.cutpaste = CutPasteUnion(color_jitter=color_jitter)

        # Few-shot augmentation (flip/rotate/color-jitter/grayscale/blur) applied
        # to base training images before CutPaste. Matches the paper's protocol
        # for expanding a tiny (1/2/4-shot) training set. Not used at eval/export
        # time, so it never affects predict/test/export outputs.
        self.few_shot_augment = FewShotAugmentation() if use_few_shot_augmentation else None

    @classmethod
    def configure_pre_processor(
        cls,
        image_size: tuple[int, int] | None = None,
    ) -> PreProcessor:
        """Configure default preprocessing for FoundAD.

        Uses ImageNet normalization matching the DINOv2 encoder's training
        distribution. Default image size is 518 (37*14) for optimal alignment
        with DINOv2 ViT-Base patch_size=14.

        Args:
            image_size: Target size as (H, W). Defaults to (518, 518).

        Returns:
            Configured PreProcessor.
        """
        image_size = image_size or (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

        data_transforms = Compose([
            Resize(image_size),
            CenterCrop(image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return PreProcessor(transform=data_transforms)

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Training step: apply CutPaste and compute manifold projection loss.

        Args:
            batch: Input batch with images.
            *args: Unused.
            **kwargs: Unused.

        Returns:
            Dict with computed loss.
        """
        del args, kwargs
        images = self.few_shot_augment(batch.image) if self.few_shot_augment is not None else batch.image
        augmented = self.cutpaste(images)
        loss = self.model(images, augmented_images=augmented)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Validation step: compute anomaly scores and maps.

        Args:
            batch: Input batch with images.
            *args: Unused.
            **kwargs: Unused.

        Returns:
            Updated batch with predictions.
        """
        del args, kwargs
        predictions = self.model(batch.image)
        return batch.update(pred_score=predictions.pred_score, anomaly_map=predictions.anomaly_map)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure AdamW optimizer for the projector parameters.

        Only the manifold projector is trained; the encoder is frozen.

        Returns:
            Optimizer configuration.
        """
        return torch.optim.AdamW(
            self.model.projector.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type.

        FoundAD is designed for few-shot settings but trains on normal samples
        only, making it ONE_CLASS in Anomalib's taxonomy.

        Returns:
            LearningType.ONE_CLASS.
        """
        return LearningType.ONE_CLASS

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return FoundAD-specific trainer arguments.

        Returns:
            Trainer configuration dict.
        """
        return {"num_sanity_val_steps": 0}
