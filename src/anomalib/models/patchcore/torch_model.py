"""PyTorch model for the PatchCore model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from omegaconf import ListConfig


from anomalib.models.components import (
    DynamicBufferModule,
    FeatureExtractor,
    KCenterGreedy,
)
from anomalib.models.components.dimensionality_reduction.random_projection import SparseRandomProjection
from anomalib.models.components.sampling.amazon_k_center_greedy import ApproximateGreedyCoresetSampler
from anomalib.models.patchcore.anomaly_map import AnomalyMapGenerator
from anomalib.pre_processing import Tiler
import logging
from torch.jit import script_if_tracing

log = logging.getLogger(__name__)


@script_if_tracing
def compute_confidence_scores(patch_scores: Tensor, max_scores: Tensor) -> Tensor:
    confidence = torch.cat([torch.index_select(patch_scores[i], 0, idx) for i, idx in enumerate(max_scores)])

    return confidence


class PatchcoreModel(DynamicBufferModule, nn.Module):
    """Patchcore Module.
    Args:
        input_size: Input size of the image.
        layers: Layers to use for feature extraction.
        backbone: Backbone to use for feature extraction.
        num_neighbors: Number of neighbors to use for anomaly score computation.
        pretrained_weights: Path to pretrained weights.
        If pretrained_weights is not None, final backbone will have pretrained_weights weights.
        Default to True.
        compress_memory_bank: If true the memory bank features are projected to a lower dimensionality following the
        Johnson-Lindenstrauss lemma.
        score_computation: Method to use for anomaly score computation either amazon or anomalib.

    """

    def __init__(
        self,
        input_size: Union[ListConfig, Tuple[int, int]],
        layers: List[str],
        backbone: Union[str, nn.Module] = "resnet18",
        pre_trained: bool = True,
        num_neighbors: int = 9,
        pretrained_weights: Optional[str] = None,
        compress_memory_bank: bool = False,
        score_computation: str = "anomalib",
    ) -> None:
        super().__init__()
        self.tiler: Optional[Tiler] = None
        self.backbone = backbone
        self.layers = layers

        input_size = tuple(input_size) if isinstance(input_size, ListConfig) else input_size
        self.input_size = input_size
        self.register_buffer("num_neighbors", torch.tensor(num_neighbors))
        self.num_neighbors: Tensor
        self.score_computation = score_computation

        # TODO: Hardcoded stuff I think for ssl?
        if pretrained_weights is not None and not isinstance(self.backbone, str):
            log.info("Loading pretrained weights")

            with open(pretrained_weights, "rb") as f:
                weights = torch.load(f)

            new_state_dict = OrderedDict()

            for key, value in weights["state_dict"].items():
                if "student" in key or "teacher" in key:
                    continue

                new_key = key.replace("model.features_extractor.", "")
                new_state_dict[new_key] = value

            self.backbone.load_state_dict(new_state_dict, strict=False)

        self.feature_extractor = FeatureExtractor(backbone=self.backbone, layers=self.layers, pre_trained=pre_trained)
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

        self.register_buffer("memory_bank", torch.Tensor())
        self.memory_bank: torch.Tensor
        self.projection_model = SparseRandomProjection(eps=0.9)
        self.compress_memory_bank = compress_memory_bank

        log.info(f"Using {self.score_computation} score computation method")

    def forward(self, input_tensor: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Return Embedding during training, or a tuple of anomaly map and anomaly score during testing.

        Steps performed:
        1. Get features from a CNN.
        2. Generate embedding based on the features.
        3. Compute anomaly map in test mode.

        Args:
            input_tensor (Tensor): Input tensor

        Returns:
            Tensor | tuple[Tensor, Tensor]: Embedding for training,
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

        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)

        if self.training:
            output = embedding
        else:
            if self.compress_memory_bank:
                embedding = self.projection_model(embedding)

            if self.score_computation == "anomalib":
                patch_scores, _ = self.nearest_neighbors(embedding=embedding, n_neighbors=self.num_neighbors)
                # Reshape patch_scores to match (batch_size, feature_dim, n_neighbours)                # TODO: Verify this is correct
                patch_scores = patch_scores.reshape(-1, width * height, patch_scores.shape[1])
                max_scores = torch.argmax(patch_scores[:, :, 0], dim=1)
                confidence = compute_confidence_scores(patch_scores, max_scores)
                weights = 1 - (torch.max(torch.exp(confidence), dim=1)[0] / torch.sum(torch.exp(confidence), dim=1))
                anomaly_score = weights * torch.max(patch_scores[:, :, 0], dim=1)[0]

                patch_scores = patch_scores[:, :, 0]
            else:
                # apply nearest neighbor search
                patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1)
                # reshape to batch dimension
                patch_scores = patch_scores.reshape((batch_size, -1))
                locations = locations.reshape((batch_size, -1))
                # compute anomaly score
                anomaly_score = self.compute_anomaly_score(patch_scores, locations, embedding)

            # reshape to w, h
            patch_scores = patch_scores.reshape((-1, 1, width, height))
            # get anomaly map
            anomaly_map = self.anomaly_map_generator(patch_scores)
            output = (anomaly_map, anomaly_score)

        return output

    def generate_embedding(self, features: dict[str, Tensor]) -> Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features: Hierarchical feature map from a CNN (ResNet18 or WideResnet)
            features: dict[str:Tensor]:

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

    def subsample_embedding(self, embedding: torch.Tensor, sampling_ratio: float, mode: str = "anomalib") -> None:
        """Subsample embedding based on coreset sampling and store to memory.

        Args:
            embedding (np.ndarray): Embedding tensor from the CNN
            sampling_ratio (float): Coreset sampling ratio
            mode (str): Sampling mode. Can be either "anomalib" or "amazon"
        """
        if mode == "anomalib":
            self.subsample_embedding_anomalib(embedding, sampling_ratio)
        elif mode == "amazon":
            self.subsample_embedding_amazon(embedding, sampling_ratio)
        else:
            raise ValueError(f"Unknown subsampling mode {mode}")

    def subsample_embedding_anomalib(self, embedding: torch.Tensor, sampling_ratio: float) -> None:
        """Subsample embedding based on coreset sampling and store to memory.

        Args:
            embedding (np.ndarray): Embedding tensor from the CNN
            sampling_ratio (float): Coreset sampling ratio
        """
        log.info("Subsampling embedding with anomalib coreset sampling")
        self.projection_model.fit(embedding)
        compressed_embedding = self.projection_model.transform(embedding)

        # Coreset Subsampling
        sampler = KCenterGreedy(sampling_ratio=sampling_ratio)
        coreset_indices = sampler.sample_coreset(compressed_embedding)
        if self.compress_memory_bank:
            self.memory_bank = compressed_embedding[coreset_indices]
        else:
            self.memory_bank = embedding[coreset_indices]

    def subsample_embedding_amazon(self, embedding: torch.Tensor, sampling_ratio: float) -> None:
        """Subsample embedding based on coreset sampling and store to memory.

        Args:
            embedding (np.ndarray): Embedding tensor from the CNN
            sampling_ratio (float): Coreset sampling ratio
        """
        log.info("Subsampling embedding with amazon coreset sampling")
        self.projection_model.fit(embedding)
        compressed_embedding = self.projection_model.transform(embedding)
        # Coreset Subsampling
        sampler = ApproximateGreedyCoresetSampler(
            percentage=sampling_ratio,
            dimension_to_project_features_to=compressed_embedding.shape[1],
            device=embedding.device,
        )

        coreset_indices = sampler._compute_greedy_coreset_indices(compressed_embedding)

        if self.compress_memory_bank:
            self.memory_bank = compressed_embedding[coreset_indices]
        else:
            self.memory_bank = embedding[coreset_indices]

    @staticmethod
    def euclidean_dist(x: Tensor, y: Tensor) -> Tensor:
        """
        Calculates pair-wise distance between row vectors in x and those in y.

        Replaces torch cdist with p=2, as cdist is not properly exported to onnx and openvino format.
        Resulting matrix is indexed by x vectors in rows and y vectors in columns.

        Args:
            x: input tensor 1
            y: input tensor 2

        Returns:
            Matrix of distances between row vectors in x and y.
        """
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)  # |x|
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)  # |y|
        # row distance can be rewritten as sqrt(|x| - 2 * x @ y.T + |y|.T)
        res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
        res = res.clamp_min_(0).sqrt_()
        return res

    def nearest_neighbors(self, embedding: Tensor, n_neighbors: int) -> tuple[Tensor, Tensor]:
        """Nearest Neighbours using brute force method and euclidean norm.

        Args:
            embedding (Tensor): Features to compare the distance with the memory bank.
            n_neighbors (int): Number of neighbors to look at

        Returns:
            Tensor: Patch scores.
            Tensor: Locations of the nearest neighbor(s).
        """
        distances = self.euclidean_dist(embedding, self.memory_bank)
        if n_neighbors == 1:
            # when n_neighbors is 1, speed up computation by using min instead of topk
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores, locations

    def compute_anomaly_score(self, patch_scores: Tensor, locations: Tensor, embedding: Tensor) -> Tensor:
        """Compute Image-Level Anomaly Score.

        Args:
            patch_scores (Tensor): Patch-level anomaly scores
            locations: Memory bank locations of the nearest neighbor for each patch location
            embedding: The feature embeddings that generated the patch scores
        Returns:
            Tensor: Image-level anomaly scores
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
        _, support_samples = self.nearest_neighbors(nn_sample, n_neighbors=self.num_neighbors)
        # 4. Find the distance of the patch features to each of the support samples
        distances = self.euclidean_dist(max_patches_features.unsqueeze(1), self.memory_bank[support_samples])
        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        # 6. Apply the weight factor to the score
        score = weights * score  # s in the paper
        return score
