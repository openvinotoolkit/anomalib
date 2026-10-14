# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model for the FoundAD implementation.

FoundAD uses a frozen DINOv2 encoder and trains a lightweight manifold projector
to project encoder features back onto the natural image manifold. Anomalies are
detected as patches where the MSE between encoder features and projector output
is high — indicating the features lie off the normal manifold.

Reference:
    Zhai et al., "Foundation Visual Encoders Are Secretly Few-Shot Anomaly
    Detectors", ICLR 2026. https://arxiv.org/abs/2510.01934

See Also:
    :class:`anomalib.models.image.foundad.lightning_model.FoundAD`:
        FoundAD Lightning model.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.models.components import GaussianBlur2d
from anomalib.models.image.foundad.components.encoder_loader import load_encoder
from anomalib.models.image.foundad.components.manifold_projector import ManifoldProjector

# DINOv2 architecture configurations
DINOV2_ARCHITECTURES: dict[str, dict[str, int]] = {
    "small": {"embed_dim": 384, "num_heads": 6},
    "base": {"embed_dim": 768, "num_heads": 12},
    "large": {"embed_dim": 1024, "num_heads": 16},
}

# Inference constants
DEFAULT_GAUSSIAN_KERNEL_SIZE = 5
DEFAULT_GAUSSIAN_SIGMA = 4.0


class FoundADModel(nn.Module):
    """FoundAD: Foundation encoder + manifold projector anomaly detection model.

    The model extracts features from a frozen DINOv2 encoder and passes them
    through a lightweight manifold projector. During training, it learns to
    project normal features back onto the manifold using MSE loss with CutPaste
    augmented supervision. During inference, high MSE indicates anomalous regions.

    Args:
        encoder_name: DINOv2 model name. Defaults to ``"dinov2_vit_base_14"``.
        pred_depth: Number of projector transformer blocks. Defaults to ``6``.
        pred_emb_dim: Projector internal dimension. Defaults to ``384``.
        n_layer: Which intermediate encoder layer to extract features from
            (counting from the last layer). Defaults to ``3``.
        top_k: Number of top anomalous patches for image-level scoring.
            Defaults to ``10``.
        dropout: Dropout rate for features before projector. Defaults to ``0.2``.
        feat_normed: Whether to L2-normalize features. Defaults to ``False``.
        use_pos_embed: Whether to use positional encoding in projector.
            Defaults to ``False``.
        image_size: Input image resolution used to size the projector's
            positional embedding when ``use_pos_embed=True``. Required to be
            consistent with the resolution the model is actually run at,
            since the encoder's own default resolution (e.g. DINOv3's 256)
            may not match. Defaults to ``None``, which falls back to the
            encoder's native/pretrained resolution.

    Raises:
        ValueError: If ``top_k`` is not a positive integer.

    Example:
        >>> model = FoundADModel(encoder_name="dinov2_vit_base_14")
        >>> images = torch.randn(2, 3, 518, 518)
        >>> model.eval()
        FoundADModel(...)
        >>> result = model(images)
        >>> result.anomaly_map.shape
        torch.Size([2, 1, 518, 518])
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
    ) -> None:
        super().__init__()

        if top_k < 1:
            msg = f"top_k must be a positive integer, got {top_k}"
            raise ValueError(msg)

        self.n_layer = n_layer
        self.top_k = top_k
        # In the original FoundAD, `feat_normed` normalizes both the extracted
        # encoder features (target/context) and the projector's own output
        # (see ManifoldProjector.forward); normalizing only one side would
        # make the training/scoring MSE compare normalized against
        # unnormalized features.
        self.feat_normed = feat_normed

        # Load frozen encoder (DINOv2/DINOv3 both loaded via timm, see encoder_loader)
        self.encoder = load_encoder(encoder_name)
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Get architecture config from the loaded encoder
        embed_dim = self.encoder.embed_dim
        num_heads = self._get_architecture_config(encoder_name)["num_heads"]
        # `patch_embed.num_patches` reflects timm's construction-time/pretrained
        # grid, not the resolution the model actually runs at (the encoder
        # supports arbitrary resolutions via `dynamic_img_size`). Derive the
        # runtime patch grid from `image_size` when known, so the positional
        # embedding (only used when `use_pos_embed=True`) is sized correctly.
        if image_size is not None:
            patches_per_side = image_size // self.encoder.patch_size
            num_patches = patches_per_side * patches_per_side
        else:
            num_patches = self.encoder.patch_embed.num_patches

        # Build manifold projector (only trainable component)
        self.projector = ManifoldProjector(
            num_patches=num_patches,
            embed_dim=embed_dim,
            predictor_embed_dim=pred_emb_dim,
            depth=pred_depth,
            num_heads=num_heads,
            use_pos_embed=use_pos_embed,
            feat_normed=feat_normed,
        )

        self.dropout = nn.Dropout(dropout)

        # Gaussian blur for anomaly map smoothing at inference
        self.gaussian_blur = GaussianBlur2d(
            sigma=DEFAULT_GAUSSIAN_SIGMA,
            channels=1,
            kernel_size=DEFAULT_GAUSSIAN_KERNEL_SIZE,
        )

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from the frozen encoder.

        Uses ``get_intermediate_layers`` to extract patch-token features
        (prefix/CLS/register tokens already stripped) from the n-th last
        encoder block. L2-normalized if ``feat_normed=True`` (matching the
        original FoundAD, which normalizes encoder features before they are
        used as either the training/scoring target or the projector's input).

        Args:
            images: Input images of shape (B, C, H, W).

        Returns:
            Patch features of shape (B, num_patches, embed_dim).
        """
        # Keep the frozen encoder in eval mode even if the parent Lightning
        # module is in training mode (Lightning's train() propagates to all
        # submodules, which would otherwise re-enable dropout/stochastic depth).
        self.encoder.eval()
        with torch.no_grad():
            features = self.encoder.get_intermediate_layers(images, n=self.n_layer)[0]
        if self.feat_normed:
            features = F.normalize(features, dim=-1)
        return features

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Pass features through the manifold projector.

        Args:
            features: Encoder features of shape (B, N, D).

        Returns:
            Projected features of shape (B, N, D).
        """
        return self.projector(features)

    def forward(
        self,
        images: torch.Tensor,
        augmented_images: torch.Tensor | None = None,
    ) -> torch.Tensor | InferenceBatch:
        """Forward pass.

        During training:
            Computes MSE loss between target features (from original images) and
            predicted features (from original or augmented images via projector).
            If ``augmented_images`` is provided, with 50% probability the projector
            receives augmented features instead of normal ones.

        During inference:
            Computes per-patch MSE, generates an anomaly map via bilinear
            interpolation, applies Gaussian smoothing, and scores images using
            top-K patch MSE values.

        Args:
            images: Input images of shape (B, C, H, W).
            augmented_images: CutPaste-augmented images for training.
                None during inference.

        Returns:
            Training: MSE loss as scalar tensor.
            Inference: ``InferenceBatch`` with ``pred_score`` and ``anomaly_map``.
        """
        # Target features (always from original images, detached)
        target = self.extract_features(images)

        if self.training:
            return self._training_forward(target, augmented_images)
        return self._inference_forward(target, images)

    def _training_forward(
        self,
        target: torch.Tensor,
        augmented_images: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute training loss.

        With 50% probability (when augmented_images is available), the projector
        receives features from augmented images. Otherwise, it receives features
        from normal images. The target is always the normal image features.

        Args:
            target: Features from normal images (B, N, D).
            augmented_images: CutPaste augmented images or None.

        Returns:
            MSE loss as scalar tensor.
        """
        # Decide whether to use augmented or normal features for context
        use_augmented = augmented_images is not None and torch.rand(1).item() < 0.5
        context = self.extract_features(augmented_images) if use_augmented else target

        predicted = self.projector(self.dropout(context))
        return F.mse_loss(target, predicted)

    def _inference_forward(self, target: torch.Tensor, images: torch.Tensor) -> InferenceBatch:
        """Compute anomaly scores and maps for inference.

        Args:
            target: Encoder features of shape (B, N, D).
            images: Original images for determining output size (B, C, H, W).

        Returns:
            InferenceBatch with pred_score and anomaly_map.
        """
        predicted = self.projector(target)
        image_size = (images.shape[2], images.shape[3])

        # Per-patch MSE: (B, N)
        patch_mse = F.mse_loss(target, predicted, reduction="none").mean(dim=2)

        # Image-level score: mean of top-K anomalous patches
        k = min(self.top_k, patch_mse.shape[1])
        pred_score = torch.topk(patch_mse, k, dim=1).values.mean(dim=1)

        # Pixel-level anomaly map. Derive the patch grid from the actual input
        # size and encoder patch size rather than assuming a square grid.
        patch_size = self.encoder.patch_size
        h, w = image_size[0] // patch_size, image_size[1] // patch_size
        anomaly_map = patch_mse.view(-1, 1, h, w)
        anomaly_map = F.interpolate(anomaly_map, size=image_size, mode="bilinear", align_corners=False)
        anomaly_map = self.gaussian_blur(anomaly_map)

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

    @staticmethod
    def _get_architecture_config(encoder_name: str) -> dict[str, int]:
        """Get architecture config from encoder name.

        Args:
            encoder_name: DINOv2 model name containing size identifier.

        Returns:
            Dict with embed_dim and num_heads.

        Raises:
            ValueError: If encoder name doesn't match any known architecture.
        """
        for arch_name, config in DINOV2_ARCHITECTURES.items():
            if arch_name in encoder_name:
                return config

        msg = f"Unsupported encoder. Name must contain one of {list(DINOV2_ARCHITECTURES.keys())}"
        raise ValueError(msg)
