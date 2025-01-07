"""PyTorch model for the PatchCore model implementation.

This module implements the PatchCore model architecture using PyTorch. PatchCore
uses a memory bank of patch features extracted from a pretrained CNN backbone to
detect anomalies.

The model stores representative patch features from normal training images and
detects anomalies by comparing test image patches against this memory bank using
nearest neighbor search.

Example:
    >>> from anomalib.models.image.patchcore.torch_model import PatchcoreModel
    >>> model = PatchcoreModel(
    ...     backbone="wide_resnet50_2",
    ...     layers=["layer2", "layer3"],
    ...     pre_trained=True,
    ...     num_neighbors=9
    ... )
    >>> input_tensor = torch.randn(32, 3, 224, 224)
    >>> output = model(input_tensor)

Paper: https://arxiv.org/abs/2106.08265

See Also:
    - :class:`anomalib.models.image.patchcore.lightning_model.Patchcore`:
        Lightning implementation of the PatchCore model
    - :class:`anomalib.models.image.patchcore.anomaly_map.AnomalyMapGenerator`:
        Anomaly map generation for PatchCore using nearest neighbor search
    - :class:`anomalib.models.components.KCenterGreedy`:
        Coreset subsampling using k-center-greedy approach
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.data import InferenceBatch
from anomalib.models.components import DynamicBufferMixin, KCenterGreedy, TimmFeatureExtractor

from .anomaly_map import AnomalyMapGenerator

if TYPE_CHECKING:
    from anomalib.data.utils.tiler import Tiler


class PatchcoreModel(DynamicBufferMixin, nn.Module):
    """PatchCore PyTorch model for anomaly detection.

    This model implements the PatchCore algorithm which uses a memory bank of patch
    features for anomaly detection. Features are extracted from a pretrained CNN
    backbone and stored in a memory bank. Anomalies are detected by comparing test
    image patches with the stored features using nearest neighbor search.

    The model works in two phases:
    1. Training: Extract and store patch features from normal training images
    2. Inference: Compare test image patches against stored features to detect
       anomalies

    Args:
        layers (Sequence[str]): Names of layers to extract features from.
        backbone (str, optional): Name of the backbone CNN network.
            Defaults to ``"wide_resnet50_2"``.
        pre_trained (bool, optional): Whether to use pre-trained backbone weights.
            Defaults to ``True``.
        num_neighbors (int, optional): Number of nearest neighbors to use.
            Defaults to ``9``.

    Example:
        >>> from anomalib.models.image.patchcore.torch_model import PatchcoreModel
        >>> model = PatchcoreModel(
        ...     backbone="wide_resnet50_2",
        ...     layers=["layer2", "layer3"],
        ...     pre_trained=True,
        ...     num_neighbors=9
        ... )
        >>> input_tensor = torch.randn(32, 3, 224, 224)
        >>> output = model(input_tensor)

    Attributes:
        tiler (Tiler | None): Optional tiler for processing large images.
        feature_extractor (TimmFeatureExtractor): CNN feature extractor.
        feature_pooler (torch.nn.AvgPool2d): Average pooling layer.
        anomaly_map_generator (AnomalyMapGenerator): Generates anomaly heatmaps.
        memory_bank (torch.Tensor): Storage for patch features from training.

    Notes:
        The model requires no optimization/backpropagation as it uses a pretrained
        backbone and nearest neighbor search.

    See Also:
        - :class:`anomalib.models.image.patchcore.lightning_model.Patchcore`:
            Lightning implementation of the PatchCore model
        - :class:`anomalib.models.image.patchcore.anomaly_map.AnomalyMapGenerator`:
            Anomaly map generation for PatchCore
        - :class:`anomalib.models.components.KCenterGreedy`:
            Coreset subsampling using k-center-greedy approach
    """

    def __init__(
        self,
        layers: Sequence[str],
        backbone: str = "wide_resnet50_2",
        pre_trained: bool = True,
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        self.backbone = backbone
        self.layers = layers
        self.num_neighbors = num_neighbors

        self.feature_extractor = TimmFeatureExtractor(
            backbone=self.backbone,
            pre_trained=pre_trained,
            layers=self.layers,
        ).eval()
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.anomaly_map_generator = AnomalyMapGenerator()

        self.register_buffer("memory_bank", torch.Tensor())
        self.memory_bank: torch.Tensor

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Process input tensor through the model.

        During training, returns embeddings extracted from the input. During
        inference, returns anomaly maps and scores computed by comparing input
        embeddings against the memory bank.

        Args:
            input_tensor (torch.Tensor): Input images of shape
                ``(batch_size, channels, height, width)``.

        Returns:
            torch.Tensor | InferenceBatch: During training, returns embeddings.
                During inference, returns ``InferenceBatch`` containing anomaly
                maps and scores.

        Example:
            >>> model = PatchcoreModel(layers=["layer1"])
            >>> input_tensor = torch.randn(32, 3, 224, 224)
            >>> output = model(input_tensor)
            >>> if model.training:
            ...     assert isinstance(output, torch.Tensor)
            ... else:
            ...     assert isinstance(output, InferenceBatch)
        """
        output_size = input_tensor.shape[-2:]
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        embedding = self.generate_embedding(features)

        if self.tiler:
            embedding = self.tiler.untile(embedding)

        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)

        if self.training:
            return embedding
        # apply nearest neighbor search
        patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1)
        # reshape to batch dimension
        patch_scores = patch_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))
        # compute anomaly score
        pred_score = self.compute_anomaly_score(patch_scores, locations, embedding)
        # reshape to w, h
        patch_scores = patch_scores.reshape((batch_size, 1, width, height))
        # get anomaly map
        anomaly_map = self.anomaly_map_generator(patch_scores, output_size)

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

    def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate embedding by concatenating multi-scale feature maps.

        Combines feature maps from different CNN layers by upsampling them to a
        common size and concatenating along the channel dimension.

        Args:
            features (dict[str, torch.Tensor]): Dictionary mapping layer names to
                feature tensors extracted from the backbone CNN.

        Returns:
            torch.Tensor: Concatenated feature embedding of shape
                ``(batch_size, num_features, height, width)``.

        Example:
            >>> features = {
            ...     "layer1": torch.randn(32, 64, 56, 56),
            ...     "layer2": torch.randn(32, 128, 28, 28)
            ... }
            >>> embedding = model.generate_embedding(features)
            >>> embedding.shape
            torch.Size([32, 192, 56, 56])
        """
        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="bilinear")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        return embeddings

    @staticmethod
    def reshape_embedding(embedding: torch.Tensor) -> torch.Tensor:
        """Reshape embedding tensor for patch-wise processing.

        Converts a 4D embedding tensor into a 2D matrix where each row represents
        a patch embedding vector.

        Args:
            embedding (torch.Tensor): Input embedding tensor of shape
                ``(batch_size, embedding_dim, height, width)``.

        Returns:
            torch.Tensor: Reshaped embedding tensor of shape
                ``(batch_size * height * width, embedding_dim)``.

        Example:
            >>> embedding = torch.randn(32, 512, 7, 7)
            >>> reshaped = PatchcoreModel.reshape_embedding(embedding)
            >>> reshaped.shape
            torch.Size([1568, 512])
        """
        embedding_size = embedding.size(1)
        return embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)

    def subsample_embedding(self, embedding: torch.Tensor, sampling_ratio: float) -> None:
        """Subsample embeddings using coreset selection.

        Uses k-center-greedy coreset subsampling to select a representative
        subset of patch embeddings to store in the memory bank.

        Args:
            embedding (torch.Tensor): Embedding tensor to subsample from.
            sampling_ratio (float): Fraction of embeddings to keep, in range (0,1].

        Example:
            >>> embedding = torch.randn(1000, 512)
            >>> model.subsample_embedding(embedding, sampling_ratio=0.1)
            >>> model.memory_bank.shape
            torch.Size([100, 512])
        """
        # Coreset Subsampling
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
        coreset = sampler.sample_coreset()
        self.memory_bank = coreset

    @staticmethod
    def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances between two sets of vectors.

        Implements an efficient matrix computation of Euclidean distances between
        all pairs of vectors in ``x`` and ``y`` without using ``torch.cdist()``.

        Args:
            x (torch.Tensor): First tensor of shape ``(n, d)``.
            y (torch.Tensor): Second tensor of shape ``(m, d)``.

        Returns:
            torch.Tensor: Distance matrix of shape ``(n, m)`` where element
                ``(i,j)`` is the distance between row ``i`` of ``x`` and row
                ``j`` of ``y``.

        Example:
            >>> x = torch.randn(100, 512)
            >>> y = torch.randn(50, 512)
            >>> distances = PatchcoreModel.euclidean_dist(x, y)
            >>> distances.shape
            torch.Size([100, 50])

        Note:
            This implementation avoids using ``torch.cdist()`` for better
            compatibility with ONNX export and OpenVINO conversion.
        """
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)  # |x|
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)  # |y|
        # row distance can be rewritten as sqrt(|x| - 2 * x @ y.T + |y|.T)
        res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
        return res.clamp_min_(0).sqrt_()

    def nearest_neighbors(self, embedding: torch.Tensor, n_neighbors: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Find nearest neighbors in memory bank for input embeddings.

        Uses brute force search with Euclidean distance to find the closest
        matches in the memory bank for each input embedding.

        Args:
            embedding (torch.Tensor): Query embeddings to find neighbors for.
            n_neighbors (int): Number of nearest neighbors to return.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - Distances to nearest neighbors (shape: ``(n, k)``)
                - Indices of nearest neighbors (shape: ``(n, k)``)
                where ``n`` is number of query embeddings and ``k`` is
                ``n_neighbors``.

        Example:
            >>> embedding = torch.randn(100, 512)
            >>> # Assuming memory_bank is already populated
            >>> scores, locations = model.nearest_neighbors(embedding, n_neighbors=5)
            >>> scores.shape, locations.shape
            (torch.Size([100, 5]), torch.Size([100, 5]))
        """
        distances = self.euclidean_dist(embedding, self.memory_bank)
        if n_neighbors == 1:
            # when n_neighbors is 1, speed up computation by using min instead of topk
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores, locations

    def compute_anomaly_score(
        self,
        patch_scores: torch.Tensor,
        locations: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute image-level anomaly scores.

        Implements the paper's weighted scoring mechanism that considers both
        the distance to nearest neighbors and the local neighborhood structure
        in the memory bank.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores.
            locations (torch.Tensor): Memory bank indices of nearest neighbors.
            embedding (torch.Tensor): Input embeddings that generated the scores.

        Returns:
            torch.Tensor: Image-level anomaly scores.

        Example:
            >>> patch_scores = torch.randn(32, 49)  # 7x7 patches
            >>> locations = torch.randint(0, 1000, (32, 49))
            >>> embedding = torch.randn(32 * 49, 512)
            >>> scores = model.compute_anomaly_score(patch_scores, locations,
            ...                                     embedding)
            >>> scores.shape
            torch.Size([32])

        Note:
            When ``num_neighbors=1``, returns the maximum patch score directly.
            Otherwise, computes weighted scores using neighborhood information.
        """
        # Don't need to compute weights if num_neighbors is 1
        if self.num_neighbors == 1:
            return patch_scores.amax(1)
        batch_size, num_patches = patch_scores.shape
        # 1. Find the patch with the largest distance to it's nearest neighbor in each image
        max_patches = torch.argmax(patch_scores, dim=1)  # indices of m^test,* in the paper
        # m^test,* in the paper
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]
        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        score = patch_scores[torch.arange(batch_size), max_patches]  # s^* in the paper
        nn_index = locations[torch.arange(batch_size), max_patches]  # indices of m^* in the paper
        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = self.memory_bank[nn_index, :]  # m^* in the paper
        # indices of N_b(m^*) in the paper
        memory_bank_effective_size = self.memory_bank.shape[0]  # edge case when memory bank is too small
        _, support_samples = self.nearest_neighbors(
            nn_sample,
            n_neighbors=min(self.num_neighbors, memory_bank_effective_size),
        )
        # 4. Find the distance of the patch features to each of the support samples
        distances = self.euclidean_dist(max_patches_features.unsqueeze(1), self.memory_bank[support_samples])
        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        # 6. Apply the weight factor to the score
        return weights * score  # s in the paper
