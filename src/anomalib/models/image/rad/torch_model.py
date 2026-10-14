# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model for RAD (Retrieval-based Anomaly Detection).

This module implements the core RAD algorithm: multi-layer feature extraction,
memory bank construction, image-level retrieval, and position-aware patch-level
anomaly scoring.

Paper: https://arxiv.org/abs/2601.22763
"""

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.data import InferenceBatch
from anomalib.models.components import (
    DynamicBufferMixin,
    GaussianBlur2d,
    TimmFeatureExtractor,
)


class RadModel(DynamicBufferMixin, nn.Module):
    """RAD PyTorch model for anomaly detection.

    Implements training-free retrieval-based anomaly detection using multi-layer
    ViT features and position-aware patch matching.

    Args:
        backbone (str): Name of the ViT backbone.
            Defaults to ``"vit_base_patch16_dinov3"``.
        pre_trained (bool): Whether to use pre-trained backbone weights.
            Defaults to ``True``.
        layers (list[int]): Block indices (0-based) to extract features from.
            Defaults to ``[3, 6, 9, 11]``.
        k_image (int): Number of nearest-neighbor training images retrieved per test image.
            The paper uses ``150`` for MVTec-AD, ``900`` for VisA and Real-IAD, and ``48``
            for 3D-ADAM. Defaults to ``150``.
        use_positional_bank (bool): Enable position-aware patch matching.
            Defaults to ``True``.
        pos_radius (int): Spatial neighborhood radius (in patch units) for
            position-aware matching. The paper uses ``1`` for MVTec-AD and 3D-ADAM,
            ``2`` for VisA, and ``0`` for Real-IAD. Defaults to ``1``.
        max_ratio (float): Fraction of highest anomaly pixels pooled for
            image-level score. ``0`` means use max. Defaults to ``0.01``.
        layer_weights (list[float] | None): Weights for each layer in score
            fusion. ``None`` means uniform. Defaults to ``None``.
        bank_dtype (torch.dtype | str | None): Storage dtype of the memory bank. ``None``
            keeps the backbone dtype (exact). ``torch.float16`` halves bank memory and
            checkpoint size and speeds up scoring, at the cost of a small (order ``1e-4``)
            perturbation of the anomaly scores. Defaults to ``None``.

    Raises:
        ValueError: If ``layers`` is empty, ``k_image`` is not positive, ``pos_radius`` is
            negative, ``max_ratio`` is outside ``[0, 1]``, or ``layer_weights`` is negative,
            sums to zero, or does not have one entry per layer.

    Example:
        >>> model = RadModel()
        >>> x = torch.randn(2, 3, 448, 448)
        >>> model.training = True
        >>> _ = model(x)
    """

    def __init__(
        self,
        backbone: str = "vit_base_patch16_dinov3",
        pre_trained: bool = True,
        layers: list[int] | None = None,
        k_image: int = 150,
        use_positional_bank: bool = True,
        pos_radius: int = 1,
        max_ratio: float = 0.01,
        layer_weights: list[float] | None = None,
        bank_dtype: torch.dtype | str | None = None,
    ) -> None:
        super().__init__()

        if layers is None:
            layers = [3, 6, 9, 11]

        if not layers:
            msg = "``layers`` must contain at least one transformer block index."
            raise ValueError(msg)
        if k_image < 1:
            msg = f"``k_image`` must be a positive integer, got {k_image}."
            raise ValueError(msg)
        if pos_radius < 0:
            msg = f"``pos_radius`` must be non-negative, got {pos_radius}."
            raise ValueError(msg)
        if not 0.0 <= max_ratio <= 1.0:
            msg = f"``max_ratio`` must lie in the range [0, 1], got {max_ratio}."
            raise ValueError(msg)

        self.layers = layers
        self.k_image = k_image
        self.use_positional_bank = use_positional_bank
        self.pos_radius = pos_radius
        self.max_ratio = max_ratio
        self.bank_dtype = getattr(torch, bank_dtype) if isinstance(bank_dtype, str) else bank_dtype

        layer_names = [f"blocks.{i}" for i in self.layers]
        try:
            self.feature_extractor = TimmFeatureExtractor(
                backbone=backbone,
                pre_trained=pre_trained,
                layers=layer_names,
                requires_grad=False,
                output_fmt="NLC",
                return_class_token=True,
                norm=True,
                dynamic_img_size=True,
            )
            self.patch_size = self.feature_extractor.patch_size
        except AttributeError as exc:
            msg = (
                "RAD requires a ViT-style timm backbone that exposes patch embeddings and supports "
                "`forward_intermediates` for token extraction (output_fmt='NLC'). "
                f"Got backbone={backbone!r}."
            )
            raise ValueError(msg) from exc

        self._layer_weights = self._normalize_layer_weights(layer_weights, len(self.layers))

        # Banks are stacked over layers and registered as buffers so that they are saved to,
        # and restored from, checkpoints together with the rest of the model. Both banks are
        # stored L2-normalized, so matching reduces to plain inner products, and ``patch_banks``
        # is position-major -- ``(layer, position, image, channel)`` -- so that retrieving the
        # neighbors of a query lands the candidates in the layout the scoring GEMM consumes.
        self.cls_banks: torch.Tensor
        self.patch_banks: torch.Tensor
        self.bank_grid: torch.Tensor
        self.register_buffer("cls_banks", torch.empty(0))
        self.register_buffer("patch_banks", torch.empty(0))
        self.register_buffer("bank_grid", torch.zeros(2, dtype=torch.long))

        # Slice plan for position-aware matching, keyed by ``(grid, pos_radius)``.
        self._neighborhood_plans: dict[tuple[int, int], list[tuple[int, int, int, int, int, int, int]]] = {}

        # Stores for accumulating features during training
        self._cls_store: list[list[torch.Tensor]] = [[] for _ in self.layers]
        self._patch_store: list[list[torch.Tensor]] = [[] for _ in self.layers]
        self._train_grid: tuple[int, int] | None = None

        # Gaussian smoothing applied to anomaly map
        self.blur = GaussianBlur2d(kernel_size=(5, 5), sigma=(1.0, 1.0), channels=1)

    @staticmethod
    def _normalize_layer_weights(layer_weights: list[float] | None, num_layers: int) -> list[float]:
        """Validate the layer fusion weights and normalise them to sum to one.

        Args:
            layer_weights (list[float] | None): Raw per-layer weights, or ``None`` for uniform.
            num_layers (int): Number of extracted layers.

        Returns:
            list[float]: Non-negative weights summing to one.

        Raises:
            ValueError: If the weights are the wrong length, negative, or sum to zero.
        """
        if layer_weights is None:
            return [1.0 / num_layers] * num_layers

        if len(layer_weights) != num_layers:
            msg = f"``layer_weights`` must have one entry per layer ({num_layers}), got {len(layer_weights)}."
            raise ValueError(msg)
        if any(weight < 0 for weight in layer_weights):
            msg = f"``layer_weights`` must be non-negative, got {layer_weights}."
            raise ValueError(msg)

        total = sum(layer_weights)
        if total <= 0:
            msg = "``layer_weights`` must sum to a positive value."
            raise ValueError(msg)

        return [weight / total for weight in layer_weights]

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Process input through the model.

        During training, extracts and stores features. During inference, computes
        anomaly maps and scores via retrieval-based matching.

        Args:
            input_tensor (torch.Tensor): Input images of shape
                ``(B, 3, H, W)``.

        Returns:
            torch.Tensor | InferenceBatch: Embeddings during training, or
                InferenceBatch with anomaly maps and scores during inference.
        """
        grid = self._patch_grid(input_tensor.shape[-2], input_tensor.shape[-1])
        features = self._extract_features(input_tensor)

        if self.training:
            self._store_features(features, grid)
            return features[0][1]  # Return CLS for Lightning compatibility

        return self._score(features, grid, input_tensor.shape[-2:])

    def _patch_grid(self, height: int, width: int) -> tuple[int, int]:
        """Return the ``(rows, cols)`` patch layout the backbone produces for an image size.

        Args:
            height (int): Input image height in pixels.
            width (int): Input image width in pixels.

        Returns:
            tuple[int, int]: Number of patch rows and columns.

        Raises:
            ValueError: If the image size is not divisible by the backbone patch size.
        """
        if height % self.patch_size or width % self.patch_size:
            msg = f"Input size ({height}, {width}) must be divisible by the backbone patch size {self.patch_size}."
            raise ValueError(msg)
        return height // self.patch_size, width // self.patch_size

    def _extract_features(
        self,
        x: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Extract multi-layer features from the backbone.

        Returns:
            List of (patch_tokens, cls_token) tuples per layer.
            patch_tokens: (B, L, C), cls_token: (B, C)
        """
        layer_names = [f"blocks.{i}" for i in self.layers]
        raw_features = self.feature_extractor(x)

        result = []
        num_prefix = self.feature_extractor.num_prefix_tokens
        for name in layer_names:
            feat = raw_features[name]  # (B, N_total, C) with prefix tokens
            cls_tok = feat[:, 0, :]  # (B, C) - CLS token
            patch_tok = feat[:, num_prefix:, :]  # (B, L, C) - patch tokens
            result.append((patch_tok, cls_tok))

        return result

    def _store_features(self, features: list[tuple[torch.Tensor, torch.Tensor]], grid: tuple[int, int]) -> None:
        """Accumulate training features on CPU so large banks do not occupy accelerator memory.

        Args:
            features (list[tuple[torch.Tensor, torch.Tensor]]): Per-layer ``(patch_tokens, cls_token)``.
            grid (tuple[int, int]): Patch grid of the current batch.

        Raises:
            ValueError: If training images do not all share the same patch grid.
        """
        if self._train_grid is None:
            self._train_grid = grid
        elif self._train_grid != grid:
            msg = (
                f"All training images must share one patch grid, got {grid} after {self._train_grid}. "
                "Use a fixed image size for the training set."
            )
            raise ValueError(msg)

        for layer_idx, (patch_tokens, cls_token) in enumerate(features):
            self._cls_store[layer_idx].append(cls_token.detach().cpu())
            self._patch_store[layer_idx].append(patch_tokens.detach().cpu())

    def build_memory_bank(self) -> None:
        """Build the memory bank from accumulated training features.

        The banks are written into pre-allocated buffers one chunk at a time, and each chunk is
        released as soon as it has been copied. Concatenating and then stacking instead would
        hold two full-size copies of the bank at once, doubling the peak memory of fitting.

        Raises:
            ValueError: If no training features have been collected.
        """
        if self._train_grid is None:
            msg = "No training features collected. Run a training epoch before building the memory bank."
            raise ValueError(msg)

        device = next(self.feature_extractor.parameters()).device
        rows, cols = self._train_grid
        num_images = sum(chunk.shape[0] for chunk in self._cls_store[0])
        channels = self._cls_store[0][0].shape[-1]
        cls_dtype = self._cls_store[0][0].dtype
        dtype = self.bank_dtype or cls_dtype

        # The CLS bank stays at full precision: it is ``num_layers x N x C`` and so negligible
        # next to the patch bank, while rounding it can reorder the image-level top-k and
        # change which training images are retrieved at all.
        cls_banks = torch.empty(len(self.layers), num_images, channels, dtype=cls_dtype, device=device)
        patch_banks = torch.empty(len(self.layers), rows * cols, num_images, channels, dtype=dtype, device=device)

        for layer_idx in range(len(self.layers)):
            offset = 0
            cls_chunks, patch_chunks = self._cls_store[layer_idx], self._patch_store[layer_idx]
            while cls_chunks:
                cls_chunk, patch_chunk = cls_chunks.pop(0), patch_chunks.pop(0)
                size = cls_chunk.shape[0]
                cls_banks[layer_idx, offset : offset + size] = F.normalize(cls_chunk, dim=-1)
                # (B, L, C) -> (L, B, C) so the bank is indexed by position first.
                patch_banks[layer_idx, :, offset : offset + size] = (
                    F.normalize(patch_chunk, dim=-1).transpose(0, 1).to(dtype)
                )
                offset += size
                del cls_chunk, patch_chunk

        self.cls_banks = cls_banks
        self.patch_banks = patch_banks
        self.bank_grid = torch.tensor(self._train_grid, dtype=torch.long, device=device)

        # Clear stores
        self._cls_store = [[] for _ in self.layers]
        self._patch_store = [[] for _ in self.layers]
        self._train_grid = None

    @torch.no_grad()
    def _score(
        self,
        features: list[tuple[torch.Tensor, torch.Tensor]],
        grid: tuple[int, int],
        output_size: tuple[int, int],
    ) -> InferenceBatch:
        """Compute anomaly maps and scores via retrieval-based matching.

        Args:
            features: Multi-layer features from test images.
            grid: Patch grid ``(rows, cols)`` of the query images.
            output_size: Original spatial size (H, W) for upsampling.

        Returns:
            InferenceBatch with anomaly_map and pred_score.

        Raises:
            ValueError: If the memory bank is empty or the token count contradicts the grid.
        """
        if self.cls_banks.numel() == 0:
            msg = "Memory bank is empty. Cannot provide anomaly scores."
            raise ValueError(msg)

        rows, cols = grid
        patch_list = [feature[0] for feature in features]  # list of (B, L, C)
        if patch_list[0].shape[1] != rows * cols:
            msg = f"Expected {rows * cols} patch tokens for grid {grid}, got {patch_list[0].shape[1]}."
            raise ValueError(msg)

        # Image-level retrieval using highest layer CLS
        cls_query = F.normalize(features[-1][1], dim=-1)  # (B, C)
        cls_bank = self.cls_banks[-1].to(cls_query.dtype)  # (N, C)
        sim_img = torch.matmul(cls_query, cls_bank.t())  # (B, N)
        k = min(self.k_image, sim_img.shape[1])
        topk_idx = torch.topk(sim_img, k, dim=-1).indices  # (B, k)

        # Multi-layer patch scoring
        patch_scores_batch = self._compute_patch_scores(patch_list, topk_idx, grid)

        # Reshape and upsample to pixel-level
        patch_maps = patch_scores_batch.view(-1, 1, rows, cols)
        anomaly_map = F.interpolate(
            patch_maps,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        # Gaussian smoothing
        anomaly_map = self.blur(anomaly_map)

        return InferenceBatch(pred_score=self._image_score(anomaly_map), anomaly_map=anomaly_map)

    def _image_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """Pool an anomaly map into an image-level score.

        Args:
            anomaly_map (torch.Tensor): Pixel-level anomaly map of shape ``(B, 1, H, W)``.

        Returns:
            torch.Tensor: Image-level anomaly scores of shape ``(B,)``.
        """
        flat_map = anomaly_map.flatten(1)
        if self.max_ratio == 0:
            return flat_map.amax(dim=1)

        top_k = max(1, int(flat_map.shape[1] * self.max_ratio))
        return flat_map.topk(top_k, dim=1).values.mean(dim=1)

    def _compute_patch_scores(
        self,
        patch_list: list[torch.Tensor],
        topk_idx: torch.Tensor,
        grid: tuple[int, int],
    ) -> torch.Tensor:
        """Compute fused multi-layer patch anomaly scores.

        Retrieval is performed one image at a time so that the candidate features scale with
        the retrieved set rather than with the batch size.

        Args:
            patch_list: Test patch features per layer, each (B, L, C).
            topk_idx: Indices of k nearest training images, (B, k).
            grid: Patch grid ``(rows, cols)``.

        Returns:
            Fused patch scores of shape (B, L).
        """
        rows, cols = grid
        batch_size = patch_list[0].shape[0]
        scores = patch_list[0].new_zeros(batch_size, rows * cols)

        if self.use_positional_bank:
            self._check_bank_grid(grid)

        # Match in the bank's dtype rather than widening it: the candidate block dwarfs the
        # query, so casting it up would undo the saving a reduced-precision bank is there for.
        match_dtype = self.patch_banks.dtype

        for image_idx in range(batch_size):
            neighbors = topk_idx[image_idx]  # (k,)
            for layer_idx, patches in enumerate(patch_list):
                query = F.normalize(patches[image_idx], dim=-1).to(match_dtype)  # (L, C)
                bank = self.patch_banks[layer_idx].index_select(1, neighbors)  # (L, k, C)

                layer_scores = (
                    self._score_positional(query, bank, grid)
                    if self.use_positional_bank
                    else self._score_global(query, bank)
                )
                scores[image_idx] += self._layer_weights[layer_idx] * layer_scores

        return scores

    @staticmethod
    def _score_global(query: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        """Score patches against all retrieved patches, without a spatial constraint.

        Args:
            query: Normalized test patches (L, C).
            bank: Normalized retrieved patches (L_bank, k, C).

        Returns:
            Anomaly scores per patch (L,).
        """
        similarity = query @ bank.reshape(-1, bank.shape[-1]).t()  # (L, L_bank*k)
        return 1.0 - similarity.amax(dim=-1)

    def _check_bank_grid(self, grid: tuple[int, int]) -> None:
        """Verify the memory bank was fitted on the same patch grid as the query.

        Args:
            grid: Patch grid ``(rows, cols)`` of the query images.

        Raises:
            ValueError: If the memory bank grid differs from the query grid.
        """
        # Grid consistency is an eager validation; extracting buffer values as Python
        # integers is not supported while Dynamo captures the model for export.
        if torch.compiler.is_compiling():
            return
        bank_rows, bank_cols = (int(size) for size in self.bank_grid)
        if (bank_rows, bank_cols) != grid:
            msg = (
                f"Position-aware matching requires the memory bank grid ({bank_rows}, {bank_cols}) to match "
                f"the query grid {grid}. Use the same image size for fitting and inference, or set "
                "``use_positional_bank=False``."
            )
            raise ValueError(msg)

    def _neighborhood_plan(self, grid: tuple[int, int]) -> list[tuple[int, int, int, int, int, int, int]]:
        """Return the cached slice plan for the position-aware neighborhood.

        Each entry ``(offset_idx, row_shift, col_shift, row_start, row_end, col_start, col_end)``
        pairs bank positions ``[row_start:row_end, col_start:col_end]`` with the query positions
        found by subtracting the shift. Entries whose shift leaves the grid are dropped.

        Args:
            grid: Patch grid ``(rows, cols)``.

        Returns:
            The slice plan for ``grid`` at the configured ``pos_radius``.
        """
        if grid not in self._neighborhood_plans:
            rows, cols = grid
            plan = []
            offset_idx = 0
            for row_shift in range(-self.pos_radius, self.pos_radius + 1):
                for col_shift in range(-self.pos_radius, self.pos_radius + 1):
                    row_start, row_end = max(0, row_shift), rows - max(0, -row_shift)
                    col_start, col_end = max(0, col_shift), cols - max(0, -col_shift)
                    if row_start < row_end and col_start < col_end:
                        plan.append((offset_idx, row_shift, col_shift, row_start, row_end, col_start, col_end))
                    offset_idx += 1
            self._neighborhood_plans[grid] = plan
        return self._neighborhood_plans[grid]

    def _score_positional(
        self,
        query: torch.Tensor,
        bank: torch.Tensor,
        grid: tuple[int, int],
    ) -> torch.Tensor:
        """Score patches against retrieved patches within a spatial neighborhood.

        The neighborhood is folded into the query rather than the bank: for every bank position
        the shifted query patches are gathered into a small ``(L, offsets, C)`` tensor, and all
        offsets are then matched in a single batched GEMM. This reads the candidate features
        once instead of once per offset, and replaces the broadcast reduction with BLAS.

        Args:
            query: Normalized test patches (L, C).
            bank: Normalized retrieved patches (L_bank, k, C).
            grid: Patch grid ``(rows, cols)``.

        Returns:
            Anomaly scores per patch (L,).
        """
        rows, cols = grid
        channels = query.shape[-1]
        plan = self._neighborhood_plan(grid)
        num_offsets = (2 * self.pos_radius + 1) ** 2

        query_grid = query.view(rows, cols, channels)
        shifted = query.new_zeros(num_offsets, rows, cols, channels)
        for offset_idx, row_shift, col_shift, row_start, row_end, col_start, col_end in plan:
            shifted[offset_idx, row_start:row_end, col_start:col_end] = query_grid[
                row_start - row_shift : row_end - row_shift,
                col_start - col_shift : col_end - col_shift,
            ]

        # (L, offsets, C) x (L, C, k) -> best match per (bank position, offset).
        similarity = torch.bmm(
            shifted.permute(1, 2, 0, 3).reshape(rows * cols, num_offsets, channels),
            bank.transpose(1, 2),
        ).amax(dim=-1)

        # Fold the bank-indexed maxima back onto the query positions they came from.
        similarity_grid = similarity.view(rows, cols, num_offsets)
        best = query.new_full((rows, cols), -1.0)
        for offset_idx, row_shift, col_shift, row_start, row_end, col_start, col_end in plan:
            target = best[row_start - row_shift : row_end - row_shift, col_start - col_shift : col_end - col_shift]
            torch.maximum(target, similarity_grid[row_start:row_end, col_start:col_end, offset_idx], out=target)

        return (1.0 - best).reshape(-1)
