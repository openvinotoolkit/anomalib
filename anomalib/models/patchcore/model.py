"""Towards Total Recall in Industrial Anomaly Detection.

Paper https://arxiv.org/abs/2106.08265.
"""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision
from kornia import gaussian_blur2d
from omegaconf import ListConfig
from torch import Tensor, nn

from anomalib.models.components import (
    AnomalyModule,
    DynamicBufferModule,
    FeatureExtractor,
    KCenterGreedy,
)
from anomalib.pre_processing import Tiler


class AnomalyMapGenerator:
    """Generate Anomaly Heatmap."""

    def __init__(
        self,
        input_size: Union[ListConfig, Tuple],
        sigma: int = 4,
    ) -> None:
        self.input_size = input_size
        self.sigma = sigma

    def compute_anomaly_map(self, patch_scores: torch.Tensor, feature_map_shape: torch.Size) -> torch.Tensor:
        """Pixel Level Anomaly Heatmap.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores
            feature_map_shape (torch.Size): 2-D feature map shape (width, height)

        Returns:
            torch.Tensor: Map of the pixel-level anomaly scores
        """
        width, height = feature_map_shape
        batch_size = len(patch_scores) // (width * height)

        anomaly_map = patch_scores[:, 0].reshape((batch_size, 1, width, height))
        anomaly_map = F.interpolate(anomaly_map, size=(self.input_size[0], self.input_size[1]))

        kernel_size = 2 * int(4.0 * self.sigma + 0.5) + 1
        anomaly_map = gaussian_blur2d(anomaly_map, (kernel_size, kernel_size), sigma=(self.sigma, self.sigma))

        return anomaly_map

    @staticmethod
    def compute_anomaly_score(patch_scores: torch.Tensor) -> torch.Tensor:
        """Compute Image-Level Anomaly Score.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores
        Returns:
            torch.Tensor: Image-level anomaly scores
        """
        max_scores = torch.argmax(patch_scores[:, 0])
        confidence = torch.index_select(patch_scores, 0, max_scores)
        weights = 1 - (torch.max(torch.exp(confidence)) / torch.sum(torch.exp(confidence)))
        score = weights * torch.max(patch_scores[:, 0])
        return score

    def __call__(self, **kwargs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns anomaly_map and anomaly_score.

        Expects `patch_scores` keyword to be passed explicitly
        Expects `feature_map_shape` keyword to be passed explicitly

        Example
        >>> anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)
        >>> map, score = anomaly_map_generator(patch_scores=numpy_array, feature_map_shape=feature_map_shape)

        Raises:
            ValueError: If `patch_scores` key is not found

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: anomaly_map, anomaly_score
        """

        if "patch_scores" not in kwargs:
            raise ValueError(f"Expected key `patch_scores`. Found {kwargs.keys()}")

        if "feature_map_shape" not in kwargs:
            raise ValueError(f"Expected key `feature_map_shape`. Found {kwargs.keys()}")

        patch_scores = kwargs["patch_scores"]
        feature_map_shape = kwargs["feature_map_shape"]

        anomaly_map = self.compute_anomaly_map(patch_scores, feature_map_shape)
        anomaly_score = self.compute_anomaly_score(patch_scores)
        return anomaly_map, anomaly_score


class PatchcoreModel(DynamicBufferModule, nn.Module):
    """Patchcore Module."""

    def __init__(
        self,
        layers: List[str],
        input_size: Tuple[int, int],
        backbone: str = "wide_resnet50_2",
        apply_tiling: bool = False,
        tile_size: Optional[Tuple[int, int]] = None,
        tile_stride: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.backbone = getattr(torchvision.models, backbone)
        self.layers = layers
        self.input_size = input_size
        self.apply_tiling = apply_tiling

        self.feature_extractor = FeatureExtractor(backbone=self.backbone(pretrained=True), layers=self.layers)
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

        if apply_tiling:
            assert tile_size is not None
            assert tile_stride is not None
            self.tiler = Tiler(tile_size, tile_stride)

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
        if self.apply_tiling:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        embedding = self.generate_embedding(features)

        if self.apply_tiling:
            embedding = self.tiler.untile(embedding)

        feature_map_shape = embedding.shape[-2:]
        embedding = self.reshape_embedding(embedding)

        if self.training:
            output = embedding
        else:
            patch_scores = self.nearest_neighbors(embedding=embedding, n_neighbors=9)
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
        print("Creating CoreSet Sampler via k-Center Greedy")
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
        print("Getting the coreset from the main embedding.")
        coreset = sampler.sample_coreset()
        print("Assigning the coreset as the memory bank.")
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


class PatchcoreLightning(AnomalyModule):
    """PatchcoreLightning Module to train PatchCore algorithm.

    Args:
        layers (List[str]): Layers used for feature extraction
        input_size (Tuple[int, int]): Input size for the model.
        tile_size (Tuple[int, int]): Tile size
        tile_stride (int): Stride for tiling
        backbone (str, optional): Pre-trained model backbone. Defaults to "resnet18".
        apply_tiling (bool, optional): Apply tiling. Defaults to False.
    """

    def __init__(self, hparams) -> None:
        super().__init__(hparams)

        self.model: PatchcoreModel = PatchcoreModel(
            layers=hparams.model.layers,
            input_size=hparams.model.input_size,
            tile_size=hparams.dataset.tiling.tile_size,
            tile_stride=hparams.dataset.tiling.stride,
            backbone=hparams.model.backbone,
            apply_tiling=hparams.dataset.tiling.apply,
        )
        self.automatic_optimization = False
        self.embeddings: List[Tensor] = []

    def configure_optimizers(self) -> None:
        """Configure optimizers.

        Returns:
            None: Do not set optimizers by returning None.
        """
        return None

    def training_step(self, batch, _batch_idx):  # pylint: disable=arguments-differ
        """Generate feature embedding of the batch.

        Args:
            batch (Dict[str, Any]): Batch containing image filename, image, label and mask
            _batch_idx (int): Batch Index

        Returns:
            Dict[str, np.ndarray]: Embedding Vector
        """
        self.model.feature_extractor.eval()
        embedding = self.model(batch["image"])

        # NOTE: `self.embedding` appends each batch embedding to
        #   store the training set embedding. We manually append these
        #   values mainly due to the new order of hooks introduced after PL v1.4.0
        #   https://github.com/PyTorchLightning/pytorch-lightning/pull/7357
        self.embeddings.append(embedding)

    def on_validation_start(self) -> None:
        """Apply subsampling to the embedding collected from the training set."""
        # NOTE: Previous anomalib versions fit subsampling at the end of the epoch.
        #   This is not possible anymore with PyTorch Lightning v1.4.0 since validation
        #   is run within train epoch.
        print("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings)

        sampling_ratio = self.hparams.model.coreset_sampling_ratio
        self.model.subsample_embedding(embeddings, sampling_ratio)

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Get batch of anomaly maps from input image batch.

        Args:
            batch (Dict[str, Any]): Batch containing image filename,
                                    image, label and mask
            _ (int): Batch Index

        Returns:
            Dict[str, Any]: Image filenames, test images, GT and predicted label/masks
        """

        anomaly_maps, anomaly_score = self.model(batch["image"])
        batch["anomaly_maps"] = anomaly_maps
        batch["pred_scores"] = anomaly_score.unsqueeze(0)

        return batch
