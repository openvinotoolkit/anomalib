# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Learning to Be a Transformer to Pinpoint Anomalies.

This module implements the L2BT model for anomaly detection as described in
`Costanzino et al. (2025) <https://ieeexplore.ieee.org/document/11048772>`_.

The model consists of:

- A pre-trained Vision Transformer teacher that extracts patch embeddings
- Two shallow student MLPs (backward_net and forward_net) that learn to match
  teacher patch embeddings
- Feature distillation between teacher and student representations
- Anomaly detection based on student ability to reconstruct teacher features

Example:
    >>> from anomalib.models.image import L2BT
    >>> from anomalib.engine import Engine
    >>> from anomalib.data import MVTecAD
    >>> datamodule = MVTecAD()
    >>> model = L2BT(
    ...     layers=(7, 11),
    ...     topk_ratio=0.001
    ... )
    >>> engine = Engine(model=model, datamodule=datamodule)
    >>> engine.fit()  # doctest: +SKIP
    >>> predictions = engine.predict()  # doctest: +SKIP

See Also:
    - :class:`L2BT`: Lightning implementation of the model
    - :class:`L2BTModel`: PyTorch implementation of the model architecture
"""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any

import torch
from torchvision.transforms.v2 import Compose, InterpolationMode, Normalize, Resize

from anomalib import LearningType
from anomalib.data.transforms import SquarePad
from anomalib.models.components import AnomalibModule
from anomalib.pre_processing import PreProcessor

from .torch_model import L2BTModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lightning.pytorch.utilities.types import STEP_OUTPUT

    from anomalib.data import Batch
    from anomalib.metrics import Evaluator
    from anomalib.post_processing import PostProcessor
    from anomalib.visualization import Visualizer

__all__ = ["L2BT"]


class L2BT(AnomalibModule):
    """Learning to Be a Transformer algorithm.

    The L2BT model consists of a pre-trained
    Vision Transformer teacher that extracts patch embeddings and two shallow
    student MLPs (backward_net and forward_net) that learn to match the teacher's
    patch embeddings. The model detects anomalies by comparing the student's
    ability to reconstruct teacher embeddings on normal images, where degradation
    indicates anomalies.

    Args:
        lr (float): Learning rate for student network optimization.
            Defaults to ``1e-4``.
        layers (Sequence[int]): Indices of Vision Transformer layers used for
            feature extraction. Must be a sequence of exactly two indices.
            Defaults to ``(7, 11)``.
        blur_w_l (int): Lower bound for blur kernel width in augmentation.
            Defaults to ``5``.
        blur_w_u (int): Upper bound for blur kernel width in augmentation.
            Defaults to ``7``.
        blur_pad_l (int): Lower bound for blur padding in augmentation.
            Defaults to ``2``.
        blur_pad_u (int): Upper bound for blur padding in augmentation.
            Defaults to ``3``.
        blur_repeats_l (int): Number of repetitions for lower blur kernel.
            Defaults to ``3``.
        blur_repeats_u (int): Number of repetitions for upper blur kernel.
            Defaults to ``5``.
        topk_ratio (float): Fraction of highest anomaly-map values to use for
            image-level anomaly scoring. Defaults to ``0.001``.
        pre_processor (PreProcessor | bool, optional): Pre-processor to transform
            input data before passing to model. If ``True``, uses default.
            Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor to generate
            predictions from model outputs. If ``True``, uses default.
            Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator to compute metrics.
            If ``True``, uses default. Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer to display results.
            If ``True``, uses default. Defaults to ``True``.

    Example:
        >>> from anomalib.models.image import L2BT
        >>> from anomalib.data import MVTecAD
        >>> from anomalib.engine import Engine
        >>> datamodule = MVTecAD()
        >>> model = L2BT(
        ...     layers=(7, 11),
        ...     topk_ratio=0.001
        ... )
        >>> engine = Engine(model=model, datamodule=datamodule)
        >>> engine.fit()  # doctest: +SKIP
        >>> predictions = engine.predict()  # doctest: +SKIP

    See Also:
        - :class:`anomalib.models.image.l2bt.torch_model.L2BTModel`:
            PyTorch implementation of the model architecture
    """

    def __init__(
        self,
        lr: float = 1e-4,
        layers: Sequence[int] = (7, 11),
        blur_w_l: int = 5,
        blur_w_u: int = 7,
        blur_pad_l: int = 2,
        blur_pad_u: int = 3,
        blur_repeats_l: int = 5,
        blur_repeats_u: int = 3,
        topk_ratio: float = 0.001,
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

        self.lr = lr
        self.model = L2BTModel(
            layers=layers,
            blur_w_l=blur_w_l,
            blur_w_u=blur_w_u,
            blur_pad_l=blur_pad_l,
            blur_pad_u=blur_pad_u,
            blur_repeats_l=blur_repeats_l,
            blur_repeats_u=blur_repeats_u,
            topk_ratio=topk_ratio,
        )

    @property
    def learning_type(self) -> LearningType:
        """Get the learning type of the model.

        Returns:
            LearningType: The model uses one-class learning.
        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_pre_processor(image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure the default pre-processor for L2BT.

        The original L2BT pipeline applies: SquarePad (edge replication) →
        Resize (bicubic interpolation) → ImageNet normalization.

        Args:
            image_size (tuple[int, int] | None, optional): Target image size for
                resizing. Defaults to ``None``. If ``None``, ``(224, 224)`` is used.

        Returns:
            PreProcessor: Configured pre-processor with the L2BT transform pipeline.
        """
        image_size = image_size or (224, 224)
        return PreProcessor(
            transform=Compose([
                SquarePad(),
                Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        )

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get required trainer arguments for the model.

        Returns:
            dict[str, Any]: Dictionary of trainer arguments (empty for L2BT as no
                special trainer configuration is required).
        """
        return {}

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a training step of L2BT.

        For each batch, teacher patch embeddings are extracted from the Vision
        Transformer, and student MLPs are trained to reconstruct these embeddings.
        Multiple loss terms are computed: main loss, middle layer loss, and final
        layer loss for comprehensive supervision.

        Args:
            batch (Batch): Input batch containing images and labels.
            args: Additional arguments (unused).
            kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Dictionary containing the loss value.
        """
        del args, kwargs  # These variables are not used.

        out = self.model(batch.image)

        loss = out["loss"]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_middle", out["loss_middle"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_loss_last", out["loss_last"], on_step=True, on_epoch=True, prog_bar=False)
        return {"loss": loss}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer with the following parameters:
                - Learning rate: as specified in the constructor (default 1e-4)
                - Optimizes parameters of both backward_net and forward_net
        """
        return torch.optim.Adam(
            params=chain(self.model.backward_net.parameters(), self.model.forward_net.parameters()),
            lr=self.lr,
        )

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a validation step of L2BT.

        Similar to training, extracts teacher patch embeddings and computes
        student reconstruction errors, generating anomaly maps for evaluation.

        Args:
            batch (Batch): Input batch containing images and labels.
            args: Additional arguments (unused).
            kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Updated batch with images, anomaly maps, labels and
                masks for evaluation.
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())
