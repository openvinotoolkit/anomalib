# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning integration for GRD-Net."""

import math
from typing import Any

import torch
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
from .pre_processor import GRDNetPreProcessor
from .tiling import rotate_image_and_roi, tile_image_and_roi
from .torch_model import GRDNetModel


class GRDNet(AnomalibModule):
    """GRD-Net anomaly detection model.

    GRD-Net uses an encoder-decoder-encoder generator, an adversarial discriminator,
    and a DRÆM segmentator. Training performs one manual optimizer step for each
    subnetwork. Region-of-interest masks supervise only the segmentator target and
    default to the full image when they are absent.

    Args:
        texture_source: Texture used to synthesize anomalies. ``"random"`` samples
            independent RGB values and ``"image"`` shifts the input tile.
        perlin_probability: Per-tile probability of applying a synthetic anomaly.
        adversarial_weight: Weight of the generator feature-matching objective.
        contextual_weight: Weight of the generator reconstruction objective.
        encoder_weight: Weight of the generator latent-consistency objective.
        learning_rate: Initial learning rate shared by all three optimizers.
        pre_processor: Pre-processor instance or flag to use the default.
        post_processor: Post-processor instance or flag to use the default.
        evaluator: Evaluator instance or flag to use the default.
        visualizer: Visualizer instance or flag to use the default.
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
    ) -> None:
        _validate_texture_source(texture_source)
        perlin_probability = _validate_numeric("perlin_probability", perlin_probability, minimum=0.0, maximum=1.0)
        adversarial_weight = _validate_numeric("adversarial_weight", adversarial_weight, minimum=0.0)
        contextual_weight = _validate_numeric("contextual_weight", contextual_weight, minimum=0.0)
        encoder_weight = _validate_numeric("encoder_weight", encoder_weight, minimum=0.0)
        learning_rate = _validate_numeric("learning_rate", learning_rate, minimum=0.0, inclusive_minimum=False)

        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        self.model = self.configure_torch_model()
        self.anomaly_generator = self.configure_anomaly_generator(texture_source, perlin_probability)
        self.generator_loss = GRDNetGeneratorLoss(
            adversarial_weight=adversarial_weight,
            contextual_weight=contextual_weight,
            encoder_weight=encoder_weight,
        )
        self.discriminator_loss = GRDNetDiscriminatorLoss()
        self.segmentator_loss = GRDNetSegmentatorLoss()
        self.contextual_loss_mean = MeanMetric()
        self.learning_rate = learning_rate
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
            factor=math.exp(-0.1),
            patience=3,
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
        image_tiles, roi_tiles = tile_image_and_roi(images, roi_masks)
        image_tiles, roi_tiles = rotate_image_and_roi(image_tiles, roi_tiles)
        perturbed_tiles, _, anomaly_masks, _ = self.anomaly_generator(image_tiles)

        discriminator_loss = self._update_discriminator(image_tiles, discriminator_optimizer)
        generator_losses = self._update_generator(image_tiles, generator_optimizer)
        segmentator_loss = self._update_segmentator(
            image_tiles,
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

    def _update_discriminator(self, images: torch.Tensor, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """Update only the adversarial discriminator."""
        self.model.generator.eval()
        self.model.discriminator.train()
        with torch.no_grad():
            _, reconstruction = self.model.generator.reconstruct(images)
        _, real_logits = self.model.discriminator(images)
        _, fake_logits = self.model.discriminator(reconstruction.detach())
        loss = self.discriminator_loss(real_logits, fake_logits)

        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss

    def _update_generator(self, images: torch.Tensor, optimizer: torch.optim.Optimizer) -> GeneratorLosses:
        """Update only the generator while keeping the discriminator frozen."""
        self.model.generator.train()
        self.model.discriminator.eval()
        requires_grad = [parameter.requires_grad for parameter in self.model.discriminator.parameters()]
        for parameter in self.model.discriminator.parameters():
            parameter.requires_grad_(requires_grad=False)
        try:
            latent, reconstruction, reconstruction_latent = self.model.generator(images)
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
            self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
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
        images: torch.Tensor,
        perturbed_images: torch.Tensor,
        anomaly_masks: torch.Tensor,
        roi_masks: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        """Update only the segmentator using detached reconstructions."""
        self.model.generator.eval()
        self.model.segmentator.train()
        with torch.no_grad():
            _, reconstruction = self.model.generator.reconstruct(images)
        logits = self.model.segmentator(torch.cat((perturbed_images, reconstruction.detach()), dim=1))
        loss = self.segmentator_loss(logits, anomaly_masks, roi_masks)

        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss

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


def _validate_texture_source(texture_source: str) -> None:
    """Validate the public texture-source option."""
    if texture_source not in {"random", "image"}:
        msg = f"Unknown texture source {texture_source!r}. Expected 'random' or 'image'."
        raise ValueError(msg)


def _validate_numeric(
    name: str,
    value: float,
    minimum: float,
    maximum: float | None = None,
    *,
    inclusive_minimum: bool = True,
) -> float:
    """Validate one finite public numeric argument."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        msg = f"{name} must be a finite number, got {value!r}."
        raise ValueError(msg)
    if (inclusive_minimum and value < minimum) or (not inclusive_minimum and value <= minimum):
        relation = "at least" if inclusive_minimum else "greater than"
        msg = f"{name} must be {relation} {minimum}, got {value!r}."
        raise ValueError(msg)
    if maximum is not None and value > maximum:
        msg = f"{name} must be at most {maximum}, got {value!r}."
        raise ValueError(msg)
    return float(value)
