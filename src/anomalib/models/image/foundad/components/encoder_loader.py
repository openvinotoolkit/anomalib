# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Loading frozen DINOv2/DINOv3 encoders for FoundAD via ``timm``.

Both DINOv2 and DINOv3 Vision Transformer backbones are hosted on ``timm``,
so FoundAD loads either family through the same ``timm.create_model`` +
``forward_intermediates`` path rather than maintaining its own from-scratch
ViT implementation. :class:`TimmDinoWrapper` normalizes the result to a
single ``get_intermediate_layers`` method (patch tokens only, prefix/CLS/
register tokens already stripped) that :class:`FoundADModel` uses regardless
of encoder family.

Example:
    >>> from anomalib.models.image.foundad.components.encoder_loader import load_encoder
    >>> encoder = load_encoder("dinov3_vit_base_16")
    >>> encoder.embed_dim
    768
"""

import logging

import timm
import torch
from torch import nn

from anomalib.models.components.feature_extractors.timm import _disable_pos_embed_antialiasing

logger = logging.getLogger(__name__)

# encoder_name -> pretrained timm model id.
TIMM_MODEL_NAMES: dict[str, str] = {
    "dinov2_vit_small_14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vit_base_14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vit_large_14": "vit_large_patch14_dinov2.lvd142m",
    "dinov3_vit_small_16": "vit_small_patch16_dinov3.lvd1689m",
    "dinov3_vit_base_16": "vit_base_patch16_dinov3.lvd1689m",
    "dinov3_vit_large_16": "vit_large_patch16_dinov3.lvd1689m",
}


class TimmDinoWrapper(nn.Module):
    """Wraps a timm DINOv2/DINOv3 model with a DINOv2-style intermediate-layer API.

    Exposes ``embed_dim``, ``patch_embed.num_patches``, and
    ``get_intermediate_layers()`` so :class:`FoundADModel` can treat any
    supported encoder identically.
    """

    def __init__(self, timm_model: nn.Module) -> None:
        super().__init__()
        self._model = timm_model
        self.embed_dim: int = timm_model.embed_dim
        self.patch_size: int = timm_model.patch_embed.patch_size[0]
        self._num_blocks: int = len(timm_model.blocks)
        self.patch_embed = _PatchEmbedProxy(timm_model)

    def get_intermediate_layers(self, x: torch.Tensor, n: int = 1) -> list[torch.Tensor]:
        """Extract patch-token features from the n last transformer blocks.

        Args:
            x: Input images (B, C, H, W).
            n: Number of layers from the end to extract (e.g. n=3 → blocks
                ``depth-3``, ``depth-2``, ``depth-1``).

        Returns:
            List of patch feature tensors, each (B, num_patches, embed_dim),
            with prefix (CLS/register) tokens already stripped.

        Raises:
            ValueError: If ``n`` is not between 1 and the encoder's block count.
        """
        if not 1 <= n <= self._num_blocks:
            msg = f"n must be between 1 and {self._num_blocks} (number of encoder blocks), got {n}"
            raise ValueError(msg)
        indices = list(range(self._num_blocks - n, self._num_blocks))
        # `forward_intermediates` reshapes patch tokens into spatial (B, C, H, W)
        # maps, which drops the prefix tokens; flatten back to (B, N, D).
        _, intermediates = self._model.forward_intermediates(x, indices=indices, norm=True)
        return [feat.flatten(2).transpose(1, 2) for feat in intermediates]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass delegating to the wrapped model."""
        return self._model(x)


class _PatchEmbedProxy:
    """Provides ``num_patches`` from a timm model's patch embedding."""

    def __init__(self, timm_model: nn.Module) -> None:
        grid = timm_model.patch_embed.grid_size
        self.num_patches: int = grid[0] * grid[1]


def load_encoder(encoder_name: str) -> TimmDinoWrapper:
    """Load a frozen, pretrained DINOv2/DINOv3 encoder.

    Args:
        encoder_name: One of the keys in :data:`TIMM_MODEL_NAMES`, e.g.
            ``"dinov2_vit_base_14"`` or ``"dinov3_vit_base_16"``.

    Returns:
        The loaded encoder, wrapped in :class:`TimmDinoWrapper`.

    Raises:
        ValueError: If ``encoder_name`` is not recognized.
    """
    timm_name = TIMM_MODEL_NAMES.get(encoder_name)
    if timm_name is None:
        msg = f"Unknown encoder '{encoder_name}'. Expected one of: {list(TIMM_MODEL_NAMES)}"
        raise ValueError(msg)

    logger.info("Loading encoder '%s' via timm ('%s')", encoder_name, timm_name)
    _disable_pos_embed_antialiasing()
    # `dynamic_img_size=True` lets the patch embedding accept resolutions other than
    # the checkpoint's native training size (verified to be numerically identical to
    # the default at that native size); without it, some encoders (e.g. DINOv2) raise
    # a hard assertion error for any other ``image_size``.
    timm_model = timm.create_model(timm_name, pretrained=True, dynamic_img_size=True)
    return TimmDinoWrapper(timm_model)
