# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Manifold Projector for FoundAD model.

Implements the lightweight ViT-based predictor that projects encoder features
back onto the natural image manifold. This is the core trainable component
of the FoundAD architecture.

The key architectural difference from a standard ViT decoder is the **global
residual skip connection**: in addition to each block's own internal
(attention/MLP) residuals, every block's output is *also* added to the
*initial* projected embedding. This matches the original paper's
``VisionTransformerPredictor`` exactly and encourages the projector to learn
small corrections to the input features rather than completely overwriting
them.

Reference:
    Zhai et al., "Foundation Visual Encoders Are Secretly Few-Shot Anomaly
    Detectors", ICLR 2026. https://arxiv.org/abs/2510.01934
    Original implementation: https://github.com/ymxlzgy/FoundAD
"""

import math

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from anomalib.models.components.dinov2.layers import Attention
from anomalib.models.image.dinomaly.components.layers import DinomalyMLP

# Default initialization standard deviation
_INIT_STD = 0.02


class ProjectorBlock(nn.Module):
    """Transformer block for the manifold projector.

    Uses standard scaled dot-product (softmax) attention, matching the
    original paper's ``VisionTransformerPredictor`` block. Softmax attention
    is numerically bounded (a convex combination of ``v``), which matters
    here because ``ManifoldProjector.forward`` adds a *global* residual skip
    on top of this block's own internal residuals, accumulating the residual
    stream ``depth + 1`` times over the forward pass (this matches the
    original paper's architecture exactly, see the module docstring). The
    same ``scaled_dot_product_attention``-based ``Attention`` is already used
    and exported for the frozen DINOv2/DINOv3 encoder elsewhere in this
    model, so OpenVINO export compatibility is preserved.

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dim to input dim.
        qkv_bias: Whether to use bias in QKV projection.
        drop: Dropout rate for MLP and projection.
        attn_drop: Dropout rate for attention weights.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = DinomalyMLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop,
            bias=True,
            apply_input_dropout=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: standard pre-norm transformer block with internal residuals.

        ``ManifoldProjector.forward`` adds an additional global residual skip
        on top of this block's output (see the module docstring).

        Args:
            x: Input tensor of shape (B, N, D).

        Returns:
            Output tensor of shape (B, N, D).
        """
        y = self.attn(self.norm1(x))
        x = x + y
        return x + self.mlp(self.norm2(x))


class ManifoldProjector(nn.Module):
    """Lightweight ViT predictor that projects features back onto the natural image manifold.

    Architecture:
        1. Linear embedding: embed_dim -> predictor_embed_dim
        2. Optional sinusoidal positional encoding
        3. N transformer blocks with global residual skip
        4. LayerNorm
        5. Linear projection: predictor_embed_dim -> embed_dim

    The global residual skip means each block's output is added to the *initial*
    projected embedding, encouraging the projector to learn manifold corrections
    rather than completely rewriting the features.

    Args:
        num_patches: Number of input patches (determines positional encoding size).
        embed_dim: Encoder feature dimension.
        predictor_embed_dim: Internal predictor dimension. Defaults to 384.
        depth: Number of transformer blocks. Defaults to 6.
        num_heads: Number of attention heads. Defaults to 12.
        mlp_ratio: MLP hidden dim ratio. Defaults to 4.0.
        use_pos_embed: Whether to use positional encoding. Defaults to False.
        feat_normed: Whether to L2-normalize output features. Defaults to False.

    Example:
        >>> projector = ManifoldProjector(num_patches=196, embed_dim=768)
        >>> features = torch.randn(2, 196, 768)
        >>> output = projector(features)
        >>> output.shape
        torch.Size([2, 196, 768])
    """

    def __init__(
        self,
        num_patches: int,
        embed_dim: int = 768,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        use_pos_embed: bool = False,
        feat_normed: bool = False,
    ) -> None:
        super().__init__()
        self.feat_normed = feat_normed

        # Dimension reduction: encoder dim -> predictor dim
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Optional positional encoding
        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.predictor_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, predictor_embed_dim),
                requires_grad=False,
            )
            trunc_normal_(self.predictor_pos_embed, std=_INIT_STD)
        else:
            self.predictor_pos_embed = None

        # Transformer blocks
        self.predictor_blocks = nn.ModuleList([
            ProjectorBlock(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
            )
            for _ in range(depth)
        ])

        # Output normalization and projection back to encoder dim
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim, eps=1e-6)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self._init_weights()
        self._fix_init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features back onto the natural image manifold.

        Args:
            x: Encoder features of shape (B, N, embed_dim).

        Returns:
            Projected features of shape (B, N, embed_dim).

        Raises:
            ValueError: If ``use_pos_embed=True`` and the number of input
                patches doesn't match the positional embedding's fixed size
                (e.g. because ``image_size`` differs from the value used to
                construct this projector).
        """
        x = self.predictor_embed(x)

        if self.use_pos_embed and self.predictor_pos_embed is not None:
            if x.shape[1] != self.predictor_pos_embed.shape[1]:
                msg = (
                    f"Input has {x.shape[1]} patches, but the positional embedding was "
                    f"built for {self.predictor_pos_embed.shape[1]} (fixed at construction "
                    "time from `num_patches`). `use_pos_embed=True` does not support "
                    "changing `image_size` after construction; use `use_pos_embed=False` "
                    "or keep `image_size` consistent."
                )
                raise ValueError(msg)
            x = x + self.predictor_pos_embed

        # Global residual: each block adds to the initial embedding
        residual = x.clone()
        for blk in self.predictor_blocks:
            x = blk(x) + residual

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)

        if self.feat_normed:
            x = torch.nn.functional.normalize(x, dim=-1)

        return x

    def _init_weights(self) -> None:
        """Initialize weights with truncated normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=_INIT_STD)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _fix_init_weight(self) -> None:
        """Rescale attention and MLP weights by layer depth.

        This prevents the residual stream from growing too large in deep networks,
        following the approach from the original FoundAD implementation.
        """
        for layer_id, layer in enumerate(self.predictor_blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
