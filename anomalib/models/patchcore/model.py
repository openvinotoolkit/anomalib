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

from anomalib.core.model import AnomalyModule
from anomalib.core.model.dynamic_module import DynamicBufferModule
from anomalib.core.model.feature_extractor import FeatureExtractor
from anomalib.data.tiler import Tiler
from anomalib.models.patchcore.utils.sampling import (
    KCenterGreedy,
    NearestNeighbors,
    SparseRandomProjection,
)


class AnomalyMapGenerator:
    """Generate Anomaly Heatmap."""

    def __init__(
        self,
        input_size: Union[ListConfig, Tuple],
        sigma: int = 4,
    ):
        self.input_size = input_size
        self.sigma = sigma

    def compute_anomaly_map(self, patch_scores: torch.Tensor) -> torch.Tensor:
        """Pixel Level Anomaly Heatmap.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores
        Returns:
            torch.Tensor: Map of the pixel-level anomaly scores
        """
        # TODO: https://github.com/openvinotoolkit/anomalib/issues/40
        batch_size = len(patch_scores) // (28 * 28)

        anomaly_map = patch_scores[:, 0].reshape((batch_size, 1, 28, 28))
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
        confidence = patch_scores[torch.argmax(patch_scores[:, 0])]
        weights = 1 - (torch.max(torch.exp(confidence)) / torch.sum(torch.exp(confidence)))
        score = weights * max(patch_scores[:, 0])
        return score

    def __call__(self, **kwargs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns anomaly_map and anomaly_score.

        Expects `patch_scores` keyword to be passed explicitly

        Example
        >>> anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)
        >>> map, score = anomaly_map_generator(patch_scores=numpy_array)

        Raises:
            ValueError: If `patch_scores` key is not found

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: anomaly_map, anomaly_score
        """

        if "patch_scores" not in kwargs:
            raise ValueError(f"Expected key `patch_scores`. Found {kwargs.keys()}")

        patch_scores = kwargs["patch_scores"]
        anomaly_map = self.compute_anomaly_map(patch_scores)
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
    ):
        super().__init__()

        self.backbone = getattr(torchvision.models, backbone)
        self.layers = layers
        self.input_size = input_size
        self.apply_tiling = apply_tiling

        self.feature_extractor = FeatureExtractor(backbone=self.backbone(pretrained=True), layers=self.layers)
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.nn_search = NearestNeighbors(n_neighbors=9)
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

        embedding = self.reshape_embedding(embedding)

        if self.training:
            output = embedding
        else:
            patch_scores, _ = self.nn_search.kneighbors(embedding)

            anomaly_map, anomaly_score = self.anomaly_map_generator(patch_scores=patch_scores)
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

    @staticmethod
    def subsample_embedding(embedding: torch.Tensor, sampling_ratio: float) -> torch.Tensor:
        """Subsample embedding based on coreset sampling.

        Args:
            embedding (np.ndarray): Embedding tensor from the CNN
            sampling_ratio (float): Coreset sampling ratio

        Returns:
            np.ndarray: Subsampled embedding whose dimensionality is reduced.
        """
        # Random projection
        random_projector = SparseRandomProjection(eps=0.9)
        random_projector.fit(embedding)

        # Coreset Subsampling
        sampler = KCenterGreedy(model=random_projector, embedding=embedding, sampling_ratio=sampling_ratio)
        coreset = sampler.sample_coreset()
        return coreset


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

    def __init__(self, hparams):
        super().__init__(hparams)

        self.model = PatchcoreModel(
            layers=hparams.model.layers,
            input_size=hparams.model.input_size,
            tile_size=hparams.dataset.tiling.tile_size,
            tile_stride=hparams.dataset.tiling.stride,
            backbone=hparams.model.backbone,
            apply_tiling=hparams.dataset.tiling.apply,
        )
        self.automatic_optimization = False

    def configure_optimizers(self):
        """Configure optimizers.

        Returns:
            None: Do not set optimizers by returning None.
        """
        return None

    def training_step(self, batch, _):  # pylint: disable=arguments-differ
        """Generate feature embedding of the batch.

        Args:
            batch (Dict[str, Any]): Batch containing image filename,
                                    image, label and mask
            _ (int): Batch Index

        Returns:
            Dict[str, np.ndarray]: Embedding Vector
        """
        self.model.feature_extractor.eval()
        embedding = self.model(batch["image"])

        return {"embedding": embedding}

    def training_epoch_end(self, outputs):
        """Concatenate batch embeddings to generate normal embedding.

        Apply coreset subsampling to the embedding set for dimensionality reduction.

        Args:
            outputs (List[Dict[str, np.ndarray]]): List of embedding vectors
        """
        embedding = torch.vstack([output["embedding"] for output in outputs])
        sampling_ratio = self.hparams.model.coreset_sampling_ratio

        embedding = self.model.subsample_embedding(embedding, sampling_ratio)

        self.model.nn_search.fit(embedding)
        self.model.memory_bank = embedding

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Get batch of anomaly maps from input image batch.

        Args:
            batch (Dict[str, Any]): Batch containing image filename,
                                    image, label and mask
            _ (int): Batch Index

        Returns:
            Dict[str, Any]: Image filenames, test images, GT and predicted label/masks
        """

        anomaly_maps, _ = self.model(batch["image"])
        batch["anomaly_maps"] = anomaly_maps

        return batch
