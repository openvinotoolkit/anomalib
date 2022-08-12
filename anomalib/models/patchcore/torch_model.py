"""PyTorch model for the PatchCore model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from anomalib.models.components import (
    DynamicBufferModule,
    FeatureExtractor,
    KCenterGreedy,
)
from anomalib.models.patchcore.anomaly_map import AnomalyMapGenerator
from anomalib.pre_processing import Tiler


class PatchcoreModel(DynamicBufferModule, nn.Module):
    """Patchcore Module."""

    def __init__(
        self,
        input_size: Tuple[int, int],
        layers: List[str],
        backbone: str = "wide_resnet50_2",
        pre_trained: bool = True,
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()
        self.tiler: Optional[Tiler] = None

        self.backbone = backbone
        self.layers = layers
        self.input_size = input_size
        self.num_neighbors = num_neighbors

        self.feature_extractor = FeatureExtractor(backbone=self.backbone, pre_trained=pre_trained, layers=self.layers)
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

        self.register_buffer("memory_bank", torch.Tensor())
        self.memory_bank: torch.Tensor

    def forward(self, input_tensor: Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return Embedding during training, or a tuple of anomaly map and anomaly score during testing.

        Steps performed:
        1. Get features from a CNN.
        2. Generate embedding based on the features.
        3. Compute anomaly map in test mode.

        Args:
            input_tensor (Tensor): Input tensor

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Embedding for training,
                anomaly map and anomaly score for testing.
        """
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        embedding = self.generate_embedding(features)

        if self.tiler:
            embedding = self.tiler.untile(embedding)

        feature_map_shape = embedding.shape[-2:]
        embedding = self.reshape_embedding(embedding)

        if self.training:
            output = embedding
        else:
            patch_scores = self.nearest_neighbors(embedding=embedding, n_neighbors=self.num_neighbors)
            anomaly_map, anomaly_score = self.anomaly_map_generator(
                patch_scores=patch_scores, feature_map_shape=feature_map_shape
            )
            output = (anomaly_map, anomaly_score)

        return output

    def generate_embedding(self, features: Dict[str, Tensor]) -> torch.Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features: Hierarchical feature map from a CNN (ResNet18 or WideResnet)
            features: Dict[str:Tensor]:

        Returns:
            Embedding vector
        """

        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        return embeddings

    @staticmethod
    def reshape_embedding(embedding: Tensor) -> Tensor:
        """Reshape Embedding.

        Reshapes Embedding to the following format:
        [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

        Args:
            embedding (Tensor): Embedding tensor extracted from CNN features.

        Returns:
            Tensor: Reshaped embedding tensor.
        """
        embedding_size = embedding.size(1)
        embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)
        return embedding

    def subsample_embedding(self, embedding: torch.Tensor, sampling_ratio: float) -> None:
        """Subsample embedding based on coreset sampling and store to memory.

        Args:
            embedding (np.ndarray): Embedding tensor from the CNN
            sampling_ratio (float): Coreset sampling ratio
        """

        # Coreset Subsampling
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
        coreset = sampler.sample_coreset()
        self.memory_bank = coreset

    def nearest_neighbors(self, embedding: Tensor, n_neighbors: int = 9) -> Tensor:
        """Nearest Neighbours using brute force method and euclidean norm.

        Args:
            embedding (Tensor): Features to compare the distance with the memory bank.
            n_neighbors (int): Number of neighbors to look at

        Returns:
            Tensor: Patch scores.
        """
        distances = torch.cdist(embedding, self.memory_bank, p=2.0)  # euclidean norm
        patch_scores, _ = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores
