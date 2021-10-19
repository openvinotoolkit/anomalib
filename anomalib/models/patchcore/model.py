"""
Towards Total Recall in Industrial Anomaly Detection
https://arxiv.org/abs/2106.08265
"""

from typing import Dict, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import ListConfig
from scipy.ndimage import gaussian_filter
from torch import Tensor, nn

from anomalib.core.model.dynamic_module import DynamicBufferModule
from anomalib.core.model.feature_extractor import FeatureExtractor
from anomalib.core.model.nearest_neighbors import NearestNeighbors
from anomalib.core.utils.random_projection import SparseRandomProjection
from anomalib.datasets.tiler import Tiler
from anomalib.models.base import SegmentationModule
from anomalib.models.patchcore.sampling_methods.kcenter_greedy import KCenterGreedy


class AnomalyMapGenerator:
    """
    Generate Anomaly Heatmap
    """

    def __init__(
        self,
        input_size: Union[ListConfig, Tuple],
        sigma: int = 4,
    ):
        self.input_size = input_size
        self.sigma = sigma

    def compute_anomaly_map(self, score_patches: np.ndarray) -> np.ndarray:
        """
        Pixel Level Anomaly Heatmap

        Args:
            score_patches (np.ndarray): [description]
        """
        anomaly_map = score_patches[:, 0].reshape((28, 28))
        anomaly_map = cv2.resize(anomaly_map, self.input_size)
        anomaly_map = gaussian_filter(anomaly_map, sigma=self.sigma)

        return anomaly_map

    @staticmethod
    def compute_anomaly_score(patch_scores: np.ndarray) -> np.ndarray:
        """
        Compute Image-Level Anomaly Score

        Args:
            patch_scores (np.ndarray): [description]
        """
        confidence = patch_scores[np.argmax(patch_scores[:, 0])]
        weights = 1 - (np.max(np.exp(confidence)) / np.sum(np.exp(confidence)))
        score = weights * max(patch_scores[:, 0])
        return score

    def __call__(self, **kwds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns anomaly_map and anomaly_score.
        Expects `patch_scores` keyword to be passed explicitly

        Example
        >>> anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)
        >>> map, score = anomaly_map_generator(patch_scores=numpy_array)

        Raises:
            ValueError: If `patch_scores` key is not found

        Returns:
            Tuple[np.ndarray, np.ndarray]: anomaly_map, anomaly_score
        """

        if "patch_scores" not in kwds:
            raise ValueError(f"Expected key `patch_scores`. Found {kwds.keys()}")

        patch_scores: np.ndarray = kwds["patch_scores"].cpu().numpy()
        anomaly_map = self.compute_anomaly_map(patch_scores)
        anomaly_score = self.compute_anomaly_score(patch_scores)
        return anomaly_map, anomaly_score


class PatchcoreModel(DynamicBufferModule, nn.Module):
    """
    Padim Module
    """

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.backbone = getattr(torchvision.models, hparams.model.backbone)
        self.layers = hparams.model.layers
        self.input_size = hparams.model.input_size

        self.feature_extractor = FeatureExtractor(backbone=self.backbone(pretrained=True), layers=self.layers)
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.nn_search = NearestNeighbors(n_neighbors=9)
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=hparams.model.input_size)

        if hparams.dataset.tiling.apply:
            self.tiler = Tiler(hparams.dataset.tiling.tile_size, hparams.dataset.tiling.stride)

        self.register_buffer("memory_bank", torch.Tensor())
        self.memory_bank: torch.Tensor

    def forward(self, input_tensor: Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get features from a CNN.
        Generate embedding based on the feautures.
        Compute anomaly map in test mode.

        Args:
            input_tensor (Tensor): Input tensor

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Embedding for training,
                anomaly map and anomaly score for testing.
        """
        if self.hparams.dataset.tiling.apply:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        embedding = self.generate_embedding(features)

        if self.hparams.dataset.tiling.apply:
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
        """
        Generate embedding from hierarchical feature map

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
        """
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
        """
        Subsample embedding based on coreset sampling

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
        selector = KCenterGreedy(embedding, 0, 0)
        selected_idx = selector.select_batch(
            model=random_projector,
            already_selected=[],
            batch_size=int(embedding.shape[0] * sampling_ratio),
        )
        embedding_coreset = embedding[selected_idx]
        return embedding_coreset


class PatchcoreLightning(SegmentationModule):
    """
    PatchcoreLightning Module to train PatchCore algorithm
    """

    def __init__(self, hparams):
        super().__init__(hparams)

        self.model = PatchcoreModel(hparams)
        self.automatic_optimization = False

    def configure_optimizers(self):
        """
        Configure optimizers

        Returns:
            None: Do not set optimizers by returning None.
        """
        return None

    def training_step(self, batch, _):  # pylint: disable=arguments-differ
        """
        Generate feature embedding of the batch.

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
        """
        Concatenate batch embeddings to generate normal embedding.
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
        """
        Load the normal embedding to use it as memory bank.
        Apply nearest neighborhood to the embedding.
        Generate the anomaly map.

        Args:
            batch (Dict[str, Any]): Batch containing image filename,
                                    image, label and mask
            _ (int): Batch Index

        Returns:
            Dict[str, Any]: Image filenames, test images, GT and predicted label/masks
        """

        anomaly_maps, _ = self.model(batch["image"])
        batch["anomaly_maps"] = torch.Tensor(anomaly_maps).unsqueeze(0).unsqueeze(0)

        return batch
