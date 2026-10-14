# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch implementation of MH-PatchCore."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.data import InferenceBatch
from anomalib.models.components import TimmFeatureExtractor

from .anomaly_map import AnomalyMapGenerator
from .components import CovarianceWhitening, MergeReduceMemoryBank, StreamingPCA

_PATCH_SIZE = 3
_PATCH_STRIDE = 1
_FEATURE_DIMENSION = 1024
_QUERY_CHUNK_SIZE = 1024


def _adaptive_avg_pool1d(
    features: torch.Tensor,
    input_length: int,
    output_length: int = _FEATURE_DIMENSION,
) -> torch.Tensor:
    """Apply adaptive average pooling with an ONNX-compatible export path.

    Args:
        features (torch.Tensor): Flattened feature vectors.
        input_length (int): Number of values in each flattened vector.
        output_length (int): Number of pooled values. Defaults to ``1024``.

    Returns:
        torch.Tensor: Pooled feature vectors.
    """
    features = features.reshape(features.shape[0], 1, input_length)
    if not torch.onnx.is_in_onnx_export():
        return F.adaptive_avg_pool1d(features, output_length).squeeze(1)

    flattened = features.squeeze(1)
    positions = torch.arange(output_length, device=flattened.device)
    starts = torch.div(positions * input_length, output_length, rounding_mode="floor")
    ends = torch.div(
        (positions + 1) * input_length + output_length - 1,
        output_length,
        rounding_mode="floor",
    )
    maximum_width = (input_length + output_length - 1) // output_length + 1
    offsets = torch.arange(maximum_width, device=flattened.device)
    indices = starts[:, None] + offsets[None, :]
    valid = indices < ends[:, None]
    indices = indices.clamp_max(input_length - 1)
    windows = flattened.index_select(1, indices.flatten()).reshape(
        flattened.shape[0],
        output_length,
        maximum_width,
    )
    return (windows * valid.to(dtype=flattened.dtype)).sum(dim=-1) / (ends - starts).to(
        dtype=flattened.dtype,
    )


def _squared_l2_distance(queries: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    query_norms = queries.square().sum(dim=1, keepdim=True)
    reference_norms = references.square().sum(dim=1, keepdim=True).T
    distances = query_norms + reference_norms - 2 * queries @ references.T
    return distances.clamp_min(0)


def _nearest_neighbors(
    queries: torch.Tensor,
    references: torch.Tensor,
    num_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    for start in range(0, len(queries), _QUERY_CHUNK_SIZE):
        distances = _squared_l2_distance(queries[start : start + _QUERY_CHUNK_SIZE], references)
        if num_neighbors == 1:
            chunk_scores, chunk_indices = distances.min(dim=1)
        else:
            chunk_scores, chunk_indices = distances.topk(k=num_neighbors, largest=False, dim=1)
        scores.append(chunk_scores)
        indices.append(chunk_indices)
    return torch.cat(scores), torch.cat(indices)


def _compute_anomaly_score(
    patch_scores: torch.Tensor,
    locations: torch.Tensor,
    embeddings: torch.Tensor,
    memory_bank: torch.Tensor,
    num_neighbors: int,
) -> torch.Tensor:
    batch_size, num_patches = patch_scores.shape
    batch_indices = torch.arange(batch_size, device=patch_scores.device)
    anchor_patch_indices = patch_scores.argmax(dim=1)
    anchor_scores = patch_scores[batch_indices, anchor_patch_indices]
    if min(num_neighbors, len(memory_bank)) <= 1:
        return anchor_scores

    anchor_queries = embeddings.reshape(batch_size, num_patches, -1)[batch_indices, anchor_patch_indices]
    anchor_bank_indices = locations[batch_indices, anchor_patch_indices]
    anchor_bank_features = memory_bank[anchor_bank_indices]
    effective_neighbors = min(num_neighbors, len(memory_bank))
    _, support_indices = _nearest_neighbors(anchor_bank_features, memory_bank, effective_neighbors)

    support_features = memory_bank[support_indices].to(dtype=torch.float64)
    anchor_queries = anchor_queries.unsqueeze(1).to(dtype=torch.float64)
    support_distances = (support_features - anchor_queries).square().sum(dim=2)
    maximum_distances = support_distances.amax(dim=1)
    denominator = torch.exp(support_distances - maximum_distances.unsqueeze(1)).sum(dim=1)
    numerator = torch.exp(anchor_scores.to(dtype=torch.float64) - maximum_distances)
    valid_denominator = (denominator > 0) & torch.isfinite(denominator)
    weights = torch.where(valid_denominator, 1 - numerator / denominator, torch.ones_like(denominator))
    return (weights.clamp(0, 1) * anchor_scores).to(dtype=patch_scores.dtype)


class MHPatchcoreModel(nn.Module):
    """Extract ordered patch embeddings for MH-PatchCore.

    Args:
        backbone (str): Name of the timm backbone. Defaults to
            ``"wide_resnet50_2.tv2_in1k"``.
        layers (Sequence[str]): Ordered backbone layers used for feature
            extraction. Defaults to ``("layer2", "layer3")``.
        pre_trained (bool): Whether to load pretrained backbone weights.
            Defaults to ``True``.
        pca_variance_ratio (float): Fraction of explained variance retained by
            incremental PCA. Defaults to ``0.99``.
        covariance_shrinkage (float): Fixed covariance shrinkage coefficient.
            Defaults to ``0.07``.
        memory_bank_size (int): Maximum number of vectors in the finalized
            memory bank. Defaults to ``1000``.
        local_coreset_size (int): Maximum number of vectors retained in each
            merge-reduce block. Defaults to ``256``.
        num_neighbors (int): Number of memory-bank neighbors used for image
            score reweighting. Defaults to ``9``.

    Raises:
        ValueError: If ``layers`` is empty, ``pca_variance_ratio`` is outside
            ``(0, 1]``, ``covariance_shrinkage`` is outside ``[0, 1]``, or a
            memory-bank size or neighbor count is not a positive integer.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2.tv2_in1k",
        layers: Sequence[str] = ("layer2", "layer3"),
        pre_trained: bool = True,
        pca_variance_ratio: float = 0.99,
        covariance_shrinkage: float = 0.07,
        memory_bank_size: int = 1000,
        local_coreset_size: int = 256,
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()
        if not layers:
            msg = "layers must contain at least one feature layer."
            raise ValueError(msg)
        if isinstance(num_neighbors, bool) or not isinstance(num_neighbors, int) or num_neighbors <= 0:
            msg = "num_neighbors must be a positive integer."
            raise ValueError(msg)

        self.backbone = backbone
        self.layers = tuple(layers)
        self.num_neighbors = num_neighbors
        self.pca = StreamingPCA(variance_ratio=pca_variance_ratio)
        self.covariance = CovarianceWhitening(shrinkage=covariance_shrinkage)
        self.memory_bank = MergeReduceMemoryBank(
            memory_bank_size=memory_bank_size,
            local_coreset_size=local_coreset_size,
        )
        self.anomaly_map_generator = AnomalyMapGenerator()
        self.feature_extractor = TimmFeatureExtractor(
            backbone=backbone,
            layers=self.layers,
            pre_trained=pre_trained,
        ).eval()
        self._unfold = nn.Unfold(
            kernel_size=_PATCH_SIZE,
            stride=_PATCH_STRIDE,
            padding=_PATCH_SIZE // 2,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Extract embeddings or return anomaly predictions.

        Training mode returns raw patch embeddings. Evaluation mode applies the
        fitted projection, whitening, memory-bank scoring, and anomaly-map path.

        Args:
            input_tensor (torch.Tensor): Image batch with shape ``[B, C, H, W]``.

        Returns:
            torch.Tensor | InferenceBatch: Raw embeddings during training, or
                anomaly scores and maps during evaluation.

        Raises:
            RuntimeError: If evaluation is requested before all fitted state is
                available.
        """
        if not self.training and not torch.compiler.is_compiling():
            # Eager-only guard: it reads buffer values, which dynamo cannot export.
            self._validate_inference_state()

        image_size = input_tensor.shape[-2:]
        features = self.feature_extractor(input_tensor)
        embeddings = self.generate_embedding(features)
        if self.training:
            return embeddings

        batch_size = input_tensor.shape[0]
        reference_grid = features[self.layers[0]].shape[-2:]
        embeddings = self.pca(embeddings)
        embeddings = self.covariance(embeddings).to(dtype=self.memory_bank.bank.dtype)
        patch_scores, locations = _nearest_neighbors(embeddings, self.memory_bank.bank, num_neighbors=1)
        patch_scores = patch_scores.reshape(batch_size, -1)
        locations = locations.reshape(batch_size, -1)
        pred_score = _compute_anomaly_score(
            patch_scores=patch_scores,
            locations=locations,
            embeddings=embeddings,
            memory_bank=self.memory_bank.bank,
            num_neighbors=self.num_neighbors,
        )
        patch_scores = patch_scores.reshape(batch_size, 1, *reference_grid)
        anomaly_map = self.anomaly_map_generator(patch_scores, image_size)
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

    def _validate_inference_state(self) -> None:
        missing_state: list[str] = []
        if not self.pca.is_fitted:
            missing_state.append("PCA")
        if not self.covariance.is_fitted:
            missing_state.append("covariance")
        if not self.memory_bank.is_fitted:
            missing_state.append("memory bank")
        if missing_state:
            msg = "MHPatchcoreModel must be fully fitted before inference; missing " + ", ".join(missing_state) + "."
            raise RuntimeError(msg)

    def generate_embedding(self, features: dict[str | int, torch.Tensor]) -> torch.Tensor:
        """Construct ordered embeddings from hierarchical feature maps.

        Args:
            features (dict[str | int, torch.Tensor]): Feature maps keyed by
                backbone layer name.

        Returns:
            torch.Tensor: Batch-major, row-major patch embeddings with 1024
                features per patch.
        """
        patched_features: list[torch.Tensor] = []
        patch_grids: list[tuple[int, int]] = []
        for layer in self.layers:
            patches, grid = self._patchify(features[layer])
            patched_features.append(patches)
            patch_grids.append(grid)

        reference_grid = patch_grids[0]
        aligned_features = [patched_features[0]]
        aligned_features.extend(
            self._align_patches(patches, source_grid, reference_grid)
            for patches, source_grid in zip(patched_features[1:], patch_grids[1:], strict=True)
        )

        aligned_features = [patches.reshape(-1, *patches.shape[-3:]) for patches in aligned_features]
        mapped_features = [
            self._map_features(patches, channels)
            for patches, channels in zip(aligned_features, self.feature_extractor.out_dims, strict=True)
        ]
        return self._aggregate_features(torch.stack(mapped_features, dim=1), len(mapped_features))

    def _patchify(self, features: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        batch_size, channels, height, width = features.shape
        patches = self._unfold(features)
        patches = patches.transpose(1, 2).reshape(batch_size, -1, channels, _PATCH_SIZE, _PATCH_SIZE)
        return patches, (height, width)

    @staticmethod
    def _align_patches(
        patches: torch.Tensor,
        source_grid: tuple[int, int],
        target_grid: tuple[int, int],
    ) -> torch.Tensor:
        if source_grid == target_grid:
            return patches

        batch_size, _, channels, patch_height, patch_width = patches.shape
        source_height, source_width = source_grid
        target_height, target_width = target_grid
        patches = patches.reshape(
            batch_size,
            source_height,
            source_width,
            channels,
            patch_height,
            patch_width,
        )
        patches = patches.permute(0, 3, 4, 5, 1, 2).reshape(-1, 1, source_height, source_width)
        patches = F.interpolate(
            patches,
            size=target_grid,
            mode="bilinear",
            align_corners=False,
        )
        patches = patches.reshape(
            batch_size,
            channels,
            patch_height,
            patch_width,
            target_height,
            target_width,
        )
        return patches.permute(0, 4, 5, 1, 2, 3).reshape(
            batch_size,
            target_height * target_width,
            channels,
            patch_height,
            patch_width,
        )

    @staticmethod
    def _map_features(patches: torch.Tensor, channels: int) -> torch.Tensor:
        return _adaptive_avg_pool1d(patches, channels * _PATCH_SIZE**2)

    @staticmethod
    def _aggregate_features(features: torch.Tensor, num_layers: int) -> torch.Tensor:
        return _adaptive_avg_pool1d(features, num_layers * _FEATURE_DIMENSION)
