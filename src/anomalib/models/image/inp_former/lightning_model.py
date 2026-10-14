# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Exploring Intrinsic Normal Prototypes within a Single Image for Universal Anomaly Detection.

INP-Former is trained on normal images using a feature-reconstruction framework based on DINOv2.

A frozen pre-trained encoder produces patch tokens, and an INP Extractor uses M learnable
query tokens with cross-attention over those patch tokens to aggregate them into M Intrinsic
Normal Prototypes (INPs) per image; an INP Coherence Loss pulls each patch feature toward
its nearest INP so the INPs reliably capture that image's normal patterns.

A Bottleneck fuses
multi-scale encoder features, and an INP-Guided Decoder reconstructs them using the
INPs as keys/values in cross-attention (with the first residual connection removed
and a ReLU on attention weights), so its output is constrained to lie in the span of
normal prototypes and anomalous queries cannot be reconstructed. A Soft Mining Loss
upweights hard-to-reconstruct tokens during training, and at test time the per-token
discrepancy between encoder features and decoder outputs is used as the anomaly
score and map.


Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import InpFormer
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = InpFormer()
    >>> engine = Engine()

    >>> engine.fit(model, datamodule=datamodule)  # doctest: +SKIP
    >>> predictions = engine.predict(model, datamodule=datamodule)  # doctest: +SKIP

Notes:
    - The model uses DINOv2 Vision Transformer as the backbone encoder
    - Features are extracted from intermediate layers (typically layers 2-9 for base models)
    - A bottleneck MLP compresses multi-layer features before reconstruction
    - Anomaly maps are computed using cosine similarity between encoder-decoder features
    - The model supports both unsupervised anomaly detection and localization

See Also:
    :class:`anomalib.models.image.inp_former.torch_model.InpFormerModel`:
        PyTorch implementation of the InpFormer model.
"""

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch.nn.init import trunc_normal_
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize

from anomalib import LearningType, PrecisionType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.models.image.dinomaly.components import StableAdamW, WarmCosineScheduler
from anomalib.models.image.inp_former.torch_model import InpFormerModel
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = 448
DEFAULT_CROP_SIZE = 392

MAX_EPOCHS_DEFAULT = 200

# Default Training hyperparameters
TRAINING_CONFIG: dict[str, Any] = {
    "optimizer": {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "weight_decay": 1e-4,
        "amsgrad": True,
        "eps": 1e-10,
    },
    "scheduler": {
        "base_value": 1e-3,
        "final_value": 1e-4,
        "warmup_iters": 100,
    },
    "trainer": {
        "gradient_clip_val": 0.1,
        "num_sanity_val_steps": 0,
        "max_epochs": MAX_EPOCHS_DEFAULT,
    },
}


class InpFormer(AnomalibModule):
    """InpFormer Lightning Module for Vision Transformer-based Anomaly Detection.

    This lightning module trains the INP-Former anomaly detection model (InpFormerModel).
    During training, the decoder learns to reconstruct normal features from Intrinsic
    Normal Prototypes (INPs) extracted from each image by an INP Extractor.
    During inference, INPs extracted from the test image guide the decoder to reconstruct
    normal regions successfully but fail on anomalous ones, and the per-token
    reconstruction error serves as the anomaly score.

    Args:
        encoder_name (str): Name of the Vision Transformer encoder to use.
            Supports DINOv2 variants (small, base, large) with different patch sizes.
            Defaults to "vit_base_patch14_reg4_dinov2".
        target_layers (list[int] | None): List of encoder layer indices to extract
            features from. If None, uses [2, 3, 4, 5, 6, 7, 8, 9] for base models
            and [4, 6, 8, 10, 12, 14, 16, 18] for large models.
        fuse_layer_encoder (list[list[int]] | None): Groupings of encoder layers
            for feature fusion. If None, uses [[0, 1, 2, 3], [4, 5, 6, 7]].
        fuse_layer_decoder (list[list[int]] | None): Groupings of decoder layers
            for feature fusion. If None, uses [[0, 1, 2, 3], [4, 5, 6, 7]].
        remove_class_token (bool): Whether to remove class token from features
            before processing. Defaults to True.
        inp_num (int): Number of Intrinsic Normal Prototypes (INPs) to extract per image.
            Defaults to 6.
        precision (str | PrecisionType, optional): Precision type for model computations.
            Can be either a string (``"float32"``, ``"float16"``) or a :class:`PrecisionType` enum value.
            Defaults to ``PrecisionType.FLOAT32``.
        pre_processor (PreProcessor | bool, optional): Pre-processor instance or
            flag to use default. Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor instance
            or flag to use default. Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator instance or flag to use
            default. Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer instance or flag to
            use default. Defaults to ``True``.

    Example:
        >>> from anomalib.data import MVTecAD
        >>> from anomalib.models import InpFormer
        >>>
        >>> # Basic usage with default parameters
        >>> model = InpFormer()
        >>>
        >>> # Custom configuration
        >>> model = InpFormer(
        ...     encoder_name="vit_large_patch14_reg4_dinov2",
        ...     inp_num=6
        ... )
        >>>
        >>> # Training with datamodule
        >>> datamodule = MVTecAD()
        >>> engine = Engine()
        >>> engine.fit(model, datamodule=datamodule)

    Note:
        The model requires significant GPU memory due to the Vision Transformer
        architecture. Consider using gradient checkpointing or smaller model
        variants for memory-constrained environments.
    """

    def __init__(
        self,
        encoder_name: str = "vit_base_patch14_reg4_dinov2",
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        remove_class_token: bool = True,
        inp_num: int = 6,
        precision: str | PrecisionType = PrecisionType.FLOAT32,
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

        self.model: InpFormerModel = InpFormerModel(
            encoder_name=encoder_name,
            inp_num=inp_num,
            target_layers=target_layers,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            remove_class_token=remove_class_token,
        )

        if isinstance(precision, str):
            precision = PrecisionType(precision.lower())

        if precision == PrecisionType.FLOAT16:
            self.model = self.model.bfloat16()
        elif precision == PrecisionType.FLOAT32:
            self.model = self.model.float()
        else:
            msg = (
                f"Unsupported precision type: {precision}. "
                f"Supported types are: {PrecisionType.FLOAT16}, {PrecisionType.FLOAT32}."
            )
            raise ValueError(msg)

        # Set the trainable parameters for the model.
        # Only the bottleneck, decoder, aggregation and prototype token parameters are trained.
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze bottleneck, decoder, aggregation and prototype token parameters
        for param in self.model.bottleneck.parameters():
            param.requires_grad = True
        for param in self.model.decoder.parameters():
            param.requires_grad = True
        for param in self.model.aggregation.parameters():
            param.requires_grad = True
        for param in self.model.prototype_token.parameters():
            param.requires_grad = True

        self.trainable_modules = torch.nn.ModuleList([
            self.model.bottleneck,
            self.model.decoder,
            self.model.aggregation,
            self.model.prototype_token,
        ])
        self._initialize_trainable_modules(self.trainable_modules)

    @classmethod
    def configure_pre_processor(
        cls,
        image_size: tuple[int, int] | None = None,
        crop_size: int | None = None,
    ) -> PreProcessor:
        """Configure the default pre-processor for InpFormer.

        Sets up image preprocessing pipeline including resizing, center cropping,
        and normalization with ImageNet statistics. The preprocessing is optimized
        for DINOv2 Vision Transformer models.

        Args:
            image_size (tuple[int, int] | None): Target size for image resizing
                as (height, width). Defaults to (448, 448).
            crop_size (int | None): Target size for center cropping (assumes square crop).
                Should be smaller than image_size. Defaults to 392.

        Returns:
            PreProcessor: Configured pre-processor with transforms for InpFormer.

        Raises:
            ValueError: If crop_size is larger than the minimum dimension of image_size.

        Note:
            The default ImageNet normalization statistics are used:
            - Mean: [0.485, 0.456, 0.406]
            - Std: [0.229, 0.224, 0.225]
        """
        crop_size = crop_size or DEFAULT_CROP_SIZE
        image_size = image_size or (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

        # Validate inputs
        if crop_size > min(image_size):
            msg = f"Crop size {crop_size} cannot be larger than image size {image_size}"
            raise ValueError(msg)

        data_transforms = Compose([
            Resize(image_size),
            CenterCrop(crop_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return PreProcessor(transform=data_transforms)

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Training step for the InpFormer model.

        Performs a single training iteration by computing feature reconstruction loss
        between encoder and decoder features. Uses cosine similarity loss with hard
        mining and adaptive weighting based on distance ratios.

        Args:
            batch (Batch): Input batch containing images and metadata.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Dictionary containing the computed loss value.

        Raises:
            ValueError: If model output doesn't contain required features during training.

        """
        del args, kwargs  # These variables are not used.
        loss = self.model(batch.image)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Validation step for the InpFormer model.

        Performs inference on the validation batch to compute anomaly scores
        and anomaly maps. The model operates in evaluation mode to generate
        predictions for anomaly detection evaluation.

        Args:
            batch (Batch): Input batch containing images and metadata.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Updated batch with pred_score (anomaly scores) and
                anomaly_map (pixel-level anomaly maps) predictions.

        Raises:
            Exception: If an error occurs during validation inference.

        Note:
            During validation, the model returns InferenceBatch with anomaly
            scores and maps computed from encoder-decoder feature comparisons.
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(pred_score=predictions.pred_score, anomaly_map=predictions.anomaly_map)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizer and learning rate scheduler for INP-Former training.

        Sets up the training configuration with frozen DINOv2 encoder and trainable
        bottleneck and decoder components. Uses StableAdamW optimizer with warm
        cosine learning rate scheduling.

        The total number of training steps is determined dynamically from the trainer
        configuration, supporting both max_steps and max_epochs settings.

        Returns:
            OptimizerLRScheduler: Tuple containing optimizer and scheduler configurations.

        Raises:
            ValueError: If neither max_epochs nor max_steps is defined.

        """
        # Determine total training steps dynamically from trainer configuration
        # Check if the trainer has valid max_epochs and max_steps set
        max_epochs = getattr(self.trainer, "max_epochs", -1)
        max_steps = getattr(self.trainer, "max_steps", -1)

        if max_epochs is None:
            max_epochs = -1
        if max_steps is None:
            max_steps = -1

        if max_epochs < 0 and max_steps < 0:
            msg = "A finite number of steps or epochs must be defined"
            raise ValueError(msg)

        if max_epochs < 0:
            # max_epochs not set, use max_steps directly
            total_steps = max_steps
        elif max_steps < 0:
            # max_steps not set, calculate from max_epochs
            total_steps = max_epochs * len(self.trainer.datamodule.train_dataloader())
        else:
            # Both are set, use the minimum (training stops at whichever comes first)
            total_steps = min(max_steps, max_epochs * len(self.trainer.datamodule.train_dataloader()))

        optimizer_config = TRAINING_CONFIG["optimizer"]
        assert isinstance(optimizer_config, dict)
        optimizer = StableAdamW([{"params": self.trainable_modules.parameters()}], **optimizer_config)

        # Create a scheduler config with dynamically determined total steps
        scheduler_config = TRAINING_CONFIG["scheduler"].copy()
        assert isinstance(scheduler_config, dict)
        scheduler_config["total_iters"] = total_steps

        lr_scheduler = WarmCosineScheduler(optimizer, **scheduler_config)

        return [optimizer], [lr_scheduler]

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        INP-Former is an unsupervised anomaly detection model that learns normal
        data patterns without requiring anomaly labels during training.

        Returns:
            LearningType: Always returns LearningType.ONE_CLASS for unsupervised learning.

        Note:
            This property may be subject to change if supervised training support
            is introduced in future versions.
        """
        return LearningType.ONE_CLASS

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return INP-Former-specific trainer arguments.

        Provides configuration arguments optimized for INP-Former training,
        excluding max_steps to allow users to set their own training duration.

        Returns:
            dict[str, Any]: Dictionary of trainer arguments with strategy
                configuration for optimal training performance. Does not include
                max_epochs so it can be set by the engine or user.

        Note:
            The max_epochs is intentionally excluded to allow user override.
        """
        trainer_config = TRAINING_CONFIG["trainer"].copy()
        assert isinstance(trainer_config, dict)
        # Remove max_epochs to allow user override
        trainer_config.pop("max_epochs", None)
        return trainer_config

    @staticmethod
    def _initialize_trainable_modules(trainable_modules: torch.nn.ModuleList) -> None:
        """Initialize trainable modules with truncated normal initialization.

        Args:
            trainable_modules: ModuleList containing modules to initialize
        """
        for m in trainable_modules.modules():
            if isinstance(m, torch.nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, torch.nn.Linear) and m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.LayerNorm):
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight, 1.0)
