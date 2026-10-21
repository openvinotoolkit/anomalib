# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning integration for GRD-Net."""

import math
from pathlib import Path
from typing import Any

import torch
from lightning import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MeanMetric
from torchvision.transforms.v2 import Normalize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.data.transforms.utils import extract_transforms_by_type
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .anomaly_generator import GRDNetAnomalyGenerator, TextureSource
from .loss import GeneratorLosses, GRDNetDiscriminatorLoss, GRDNetGeneratorLoss, GRDNetSegmentatorLoss
from .pre_processor import GRDNetPreProcessor, _GRDNetROIGuard
from .tiling import rotate_image_and_roi, tile_image_and_roi
from .torch_model import GRDNetModel
from .validation import _validate_integer, _validate_numeric, _validate_texture_source


class GRDNet(AnomalibModule):
    """GRD-Net anomaly detection model.

    GRD-Net uses an encoder-decoder-encoder generator, an adversarial discriminator,
    and a DRÆM segmentator. Training performs one manual optimizer step for each
    subnetwork. Region-of-interest masks supervise only the segmentator target and
    default to the full image when they are absent.

    Args:
        texture_source (TextureSource): Texture used to synthesize anomalies. ``"random"`` samples
            independent RGB values and ``"image"`` shifts the input tile. Defaults to ``"random"``.
        perlin_probability (float): Per-tile probability of a synthetic anomaly. Defaults to ``0.75``.
        adversarial_weight (float): Generator feature-matching weight. Defaults to ``1.0``.
        contextual_weight (float): Generator L1 plus one-minus-SSIM weight. Defaults to ``50.0``.
        encoder_weight (float): Generator latent-consistency weight. Defaults to ``1.0``.
        learning_rate (float): Initial learning rate of all three optimizers. Defaults to ``1e-4``.
        pre_processor (PreProcessor | bool): Instance or default flag. Defaults to ``True``.
        post_processor (PostProcessor | bool): Instance or default flag. Defaults to ``True``.
        evaluator (Evaluator | bool): Instance or default flag. Defaults to ``True``.
        visualizer (Visualizer | bool): Instance or default flag. Defaults to ``True``.
        contextual_l1_weight (float): Inner L1 reconstruction coefficient. Defaults to ``1.0``.
        contextual_ssim_weight (float): Inner one-minus-SSIM coefficient. Defaults to ``1.0``.
        generator_scheduler_decay (float): Positive exponent for the generator plateau reduction
            factor ``exp(-decay)``. Defaults to ``0.1``.
        generator_scheduler_patience (int): Generator plateau patience in epochs. Defaults to ``3``.
        rotation_degrees (float): Symmetric bound for paired tile rotation. Defaults to ``90.0``.
        smoothing_kernel_size (int): Positive odd inference pooling kernel. ``1`` disables smoothing.
            Defaults to ``21``.

    Raises:
        ValueError: If a texture source is unsupported, a numeric parameter is a boolean or nonfinite,
            a probability is outside ``[0, 1]``, a weight is negative, both inner contextual weights are zero,
            an integer parameter is invalid, or a learning rate or scheduler factor is invalid.

    Example:
        >>> from anomalib.models import GRDNet
        >>> model = GRDNet(texture_source="image")
        >>> model.learning_type
        <LearningType.ONE_CLASS: 'one_class'>
    """

    def __init__(
        self,
        texture_source: TextureSource = "random",
        perlin_probability: float = 0.75,
        adversarial_weight: float = 1.0,
        contextual_weight: float = 50.0,
        encoder_weight: float = 1.0,
        learning_rate: float = 1e-4,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
        *,
        contextual_l1_weight: float = 1.0,
        contextual_ssim_weight: float = 1.0,
        generator_scheduler_decay: float = 0.1,
        generator_scheduler_patience: int = 3,
        rotation_degrees: float = 90.0,
        smoothing_kernel_size: int = 21,
    ) -> None:
        _validate_texture_source(texture_source)
        perlin_probability = _validate_numeric("perlin_probability", perlin_probability, minimum=0.0, maximum=1.0)
        adversarial_weight = _validate_numeric("adversarial_weight", adversarial_weight, minimum=0.0)
        contextual_weight = _validate_numeric("contextual_weight", contextual_weight, minimum=0.0)
        encoder_weight = _validate_numeric("encoder_weight", encoder_weight, minimum=0.0)
        learning_rate = _validate_numeric("learning_rate", learning_rate, minimum=0.0, inclusive_minimum=False)
        contextual_l1_weight = _validate_numeric("contextual_l1_weight", contextual_l1_weight, minimum=0.0)
        contextual_ssim_weight = _validate_numeric("contextual_ssim_weight", contextual_ssim_weight, minimum=0.0)
        if contextual_l1_weight == 0 and contextual_ssim_weight == 0:
            msg = "contextual_l1_weight and contextual_ssim_weight must not both be zero."
            raise ValueError(msg)
        generator_scheduler_decay = _validate_numeric(
            "generator_scheduler_decay",
            generator_scheduler_decay,
            minimum=0.0,
            inclusive_minimum=False,
        )
        if not 0 < math.exp(-generator_scheduler_decay) < 1:
            msg = "generator_scheduler_decay must yield an exp(-decay) factor strictly between zero and one."
            raise ValueError(msg)
        generator_scheduler_patience = _validate_integer(
            "generator_scheduler_patience",
            generator_scheduler_patience,
            minimum=0,
        )
        rotation_degrees = _validate_numeric("rotation_degrees", rotation_degrees, minimum=0.0)
        smoothing_kernel_size = _validate_integer("smoothing_kernel_size", smoothing_kernel_size, minimum=1)
        if smoothing_kernel_size % 2 == 0:
            msg = "smoothing_kernel_size must be a positive odd integer."
            raise ValueError(msg)

        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        self.model = self.configure_torch_model()
        self.model.smoothing_kernel_size = smoothing_kernel_size
        self.anomaly_generator = self.configure_anomaly_generator(texture_source, perlin_probability)
        self.generator_loss = GRDNetGeneratorLoss(
            adversarial_weight=adversarial_weight,
            contextual_weight=contextual_weight,
            encoder_weight=encoder_weight,
            contextual_l1_weight=contextual_l1_weight,
            contextual_ssim_weight=contextual_ssim_weight,
        )
        self.discriminator_loss = GRDNetDiscriminatorLoss()
        self.segmentator_loss = GRDNetSegmentatorLoss()
        self.contextual_loss_mean = MeanMetric()
        self.learning_rate = learning_rate
        self.generator_scheduler_decay = generator_scheduler_decay
        self.generator_scheduler_patience = generator_scheduler_patience
        self.rotation_degrees = rotation_degrees
        self.automatic_optimization = False

    @staticmethod
    def configure_torch_model() -> GRDNetModel:
        """Create the canonical GRD-Net Torch model.

        Returns:
            Canonical GRD-Net Torch model.
        """
        return GRDNetModel()

    @staticmethod
    def configure_anomaly_generator(
        texture_source: TextureSource,
        probability: float,
    ) -> GRDNetAnomalyGenerator:
        """Create the synthetic anomaly generator.

        Args:
            texture_source: Texture used inside synthetic anomaly masks.
            probability: Per-tile probability of applying an anomaly.

        Returns:
            Configured GRD-Net anomaly generator.
        """
        return GRDNetAnomalyGenerator(texture_source=texture_source, probability=probability)

    @classmethod
    def configure_pre_processor(cls, image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Create the default joint image and ROI pre-processor.

        Args:
            image_size: Unused compatibility argument. GRD-Net uses a fixed
                ``256 x 256`` input size.

        Returns:
            GRD-Net pre-processor with fixed canonical geometry.
        """
        del image_size
        return GRDNetPreProcessor()

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return GRD-Net-specific Trainer arguments.

        Returns:
            Trainer overrides that disable automatic clipping and sanity validation.
        """
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    def configure_callbacks(self) -> list[Callback]:
        """Place the ROI compatibility guard before preprocessing.

        Returns:
            ROI guard followed by the ordinary model component callbacks.
        """
        callbacks = super().configure_callbacks()
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]
        return [_GRDNetROIGuard(), *callbacks]

    def to_onnx(
        self,
        export_root: Path | str,
        model_file_name: str = "model",
        input_size: tuple[int, int] | None = None,
        **kwargs: Any,  # noqa: ANN401  # passthrough to torch.onnx.export
    ) -> Path:
        """Export image-only inference with a dynamic batch dimension.

        Args:
            export_root: Output directory.
            model_file_name: Exported model name.
            input_size: Fixed input height and width, or inherited dynamic spatial behavior.
            **kwargs: Options forwarded to the inherited ONNX exporter.

        Returns:
            Path to the exported ONNX model.
        """
        if input_size is not None and "dynamic_shapes" not in kwargs and "dynamic_axes" not in kwargs:
            # Avoid CUDA BatchNorm guards on the ninefold internal tile batch.
            kwargs["dynamic_shapes"] = ({0: torch.export.Dim.DYNAMIC},)
        return super().to_onnx(export_root, model_file_name, input_size, **kwargs)

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type.

        Returns:
            One-class learning type.
        """
        return LearningType.ONE_CLASS

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.ReduceLROnPlateau]]:
        """Configure the discriminator, generator, and segmentator optimizers.

        Returns:
            Three Adam optimizers in training-phase order and the generator scheduler.
        """
        discriminator_optimizer = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.999),
        )
        generator_optimizer = torch.optim.Adam(
            self.model.generator.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.999),
        )
        segmentator_optimizer = torch.optim.Adam(
            self.model.segmentator.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            generator_optimizer,
            mode="min",
            factor=math.exp(-self.generator_scheduler_decay),
            patience=self.generator_scheduler_patience,
            threshold=1e-4,
            threshold_mode="abs",
        )
        return (
            [discriminator_optimizer, generator_optimizer, segmentator_optimizer],
            [scheduler],
        )

    def on_train_start(self) -> None:
        """Reject input normalization before training.

        Raises:
            ValueError: If the configured pre-processor contains normalization.
        """
        if self.pre_processor and extract_transforms_by_type(self.pre_processor.transform, Normalize):
            msg = "Transforms for GRD-Net must not contain Normalize."
            raise ValueError(msg)

    def training_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        """Perform the three GRD-Net optimization phases.

        Args:
            batch: Training batch containing images and optional ROI masks.
            batch_idx: Index of the current batch.

        Returns:
            Loss values for the discriminator, generator, and segmentator phases.
        """
        del batch_idx
        discriminator_optimizer, generator_optimizer, segmentator_optimizer = self.optimizers()

        images = batch.image
        if images is None:
            msg = "GRD-Net training requires an image batch."
            raise ValueError(msg)
        roi_masks = self._roi_masks(batch)
        image_tiles, roi_tiles = tile_image_and_roi(
            images,
            roi_masks,
            tile_size=self.model.tile_size,
            stride=self.model.stride,
        )
        image_tiles, roi_tiles = rotate_image_and_roi(image_tiles, roi_tiles, rotation_degrees=self.rotation_degrees)
        perturbed_tiles, _, anomaly_masks, _ = self.anomaly_generator(image_tiles)

        discriminator_loss = self._update_discriminator(image_tiles, perturbed_tiles, discriminator_optimizer)
        generator_losses = self._update_generator(image_tiles, perturbed_tiles, generator_optimizer)
        segmentator_loss = self._update_segmentator(
            perturbed_tiles,
            anomaly_masks,
            roi_tiles,
            segmentator_optimizer,
        )

        self.contextual_loss_mean.update(generator_losses.contextual.detach())
        reported_total = generator_losses.total.detach() + segmentator_loss.detach()
        self.log_dict(
            {
                "train_discriminator_loss": discriminator_loss.detach(),
                "train_generator_loss": generator_losses.total.detach(),
                "train_segmentator_loss": segmentator_loss.detach(),
                "train_loss": reported_total,
            },
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=images.shape[0],
        )
        return {
            "discriminator_loss": discriminator_loss.detach(),
            "generator_loss": generator_losses.total.detach(),
            "segmentator_loss": segmentator_loss.detach(),
            "loss": reported_total,
        }

    def _update_discriminator(
        self,
        images: torch.Tensor,
        perturbed_images: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        """Update the discriminator using clean images and denoised reconstructions."""
        self.model.generator.eval()
        self.model.discriminator.train()
        with torch.no_grad():
            _, reconstruction = self.model.generator.reconstruct(perturbed_images)
        _, real_logits = self.model.discriminator(images)
        _, fake_logits = self.model.discriminator(reconstruction.detach())
        loss = self.discriminator_loss(real_logits, fake_logits)

        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss

    def _update_generator(
        self,
        images: torch.Tensor,
        perturbed_images: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> GeneratorLosses:
        """Train the generator to reconstruct clean images from perturbed inputs."""
        self.model.generator.train()
        self.model.discriminator.eval()
        requires_grad = [parameter.requires_grad for parameter in self.model.discriminator.parameters()]
        for parameter in self.model.discriminator.parameters():
            parameter.requires_grad_(requires_grad=False)
        try:
            latent, reconstruction, reconstruction_latent = self.model.generator(perturbed_images)
            with torch.no_grad():
                real_features, _ = self.model.discriminator(images)
            fake_features, _ = self.model.discriminator(reconstruction)
            losses = self.generator_loss(
                images,
                reconstruction,
                latent,
                reconstruction_latent,
                real_features,
                fake_features,
            )

            optimizer.zero_grad()
            self.manual_backward(losses.total)
            optimizer.step()
        finally:
            for parameter, original_requires_grad in zip(
                self.model.discriminator.parameters(),
                requires_grad,
                strict=True,
            ):
                parameter.requires_grad_(original_requires_grad)
        return losses

    def _update_segmentator(
        self,
        perturbed_images: torch.Tensor,
        anomaly_masks: torch.Tensor,
        roi_masks: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        """Update only the segmentator using detached reconstructions."""
        self.model.generator.eval()
        self.model.segmentator.train()
        with torch.no_grad():
            _, reconstruction = self.model.generator.reconstruct(perturbed_images)
        logits = self.model.segmentator(torch.cat((perturbed_images, reconstruction.detach()), dim=1))
        loss = self.segmentator_loss(logits, anomaly_masks, roi_masks)

        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Clip generator gradients after Lightning's precision plugin unscales them.

        Args:
            optimizer: Optimizer about to perform its step.
        """
        if optimizer is self.optimizers(use_pl_optimizer=False)[1]:
            torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), max_norm=1.0)

    def on_train_epoch_end(self) -> None:
        """Step the generator scheduler using mean contextual loss."""
        contextual_loss = self.contextual_loss_mean.compute()
        scheduler = self.lr_schedulers()
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            msg = "GRD-Net requires one ReduceLROnPlateau generator scheduler."
            raise TypeError(msg)
        scheduler.step(contextual_loss)
        self.log("train_contextual_loss", contextual_loss, logger=True)
        self.contextual_loss_mean.reset()

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Predict anomaly scores and maps for a validation batch.

        Args:
            batch: Validation batch.
            args: Additional positional arguments, which are unused.
            kwargs: Additional keyword arguments, which are unused.

        Returns:
            Batch updated with GRD-Net predictions.
        """
        del args, kwargs
        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @staticmethod
    def _roi_masks(batch: Batch) -> torch.Tensor:
        """Return a validated single-channel float ROI batch."""
        images = batch.image
        if images is None:
            msg = "GRD-Net training requires an image batch."
            raise ValueError(msg)
        roi_masks = getattr(batch, "roi_mask", None)
        if roi_masks is None:
            return torch.ones(
                (images.shape[0], 1, *images.shape[-2:]),
                device=images.device,
                dtype=images.dtype,
            )
        if roi_masks.ndim == 3:
            roi_masks = roi_masks.unsqueeze(1)
        if (
            roi_masks.ndim != 4
            or roi_masks.shape[1] != 1
            or roi_masks.shape[0] != images.shape[0]
            or roi_masks.shape[-2:] != images.shape[-2:]
        ):
            msg = (
                "Image and ROI batch/spatial dimensions must match before GRD-Net training, "
                f"got {tuple(images.shape)} and {tuple(roi_masks.shape)}."
            )
            raise ValueError(msg)
        return roi_masks.to(device=images.device, dtype=images.dtype)
