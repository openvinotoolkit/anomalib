"""PyTorch model for the PaDiM model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from random import sample

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from anomalib.models.components import FeatureExtractor, MultiVariateGaussian
from anomalib.models.components.feature_extractors import dryrun_find_featuremap_dims
from anomalib.models.padim.anomaly_map import AnomalyMapGenerator
from anomalib.pre_processing import Tiler

# defaults from the paper
_N_FEATURES_DEFAULTS = {
    "resnet18": 100,
    "wide_resnet50_2": 550,
}


def _deduce_dims(
    feature_extractor: FeatureExtractor, input_size: tuple[int, int], layers: list[str]
) -> tuple[int, int]:
    """Run a dry run to deduce the dimensions of the extracted features.

    Important: `layers` is assumed to be ordered and the first (layers[0])
                is assumed to be the layer with largest resolution.

    Returns:
        tuple[int, int]: Dimensions of the extracted features: (n_dims_original, n_patches)
    """
    dimensions_mapping = dryrun_find_featuremap_dims(feature_extractor, input_size, layers)

    # the first layer in `layers` has the largest resolution
    first_layer_resolution = dimensions_mapping[layers[0]]["resolution"]
    n_patches = torch.tensor(first_layer_resolution).prod().int().item()

    # the original embedding size is the sum of the channels of all layers
    n_features_original = sum(dimensions_mapping[layer]["num_features"] for layer in layers)  # type: ignore

    return n_features_original, n_patches


class PadimModel(nn.Module):
    """Padim Module.

    Args:
        input_size (tuple[int, int]): Input size for the model.
        layers (list[str]): Layers used for feature extraction
        backbone (str, optional): Pre-trained model backbone. Defaults to "resnet18".
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        n_features (int, optional): Number of features to retain in the dimension reduction step.
                                Default values from the paper are available for: resnet18 (100), wide_resnet50_2 (550).
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        layers: list[str],
        backbone: str = "resnet18",
        pre_trained: bool = True,
        n_features: int | None = None,
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        self.backbone = backbone
        self.layers = layers
        self.feature_extractor = FeatureExtractor(backbone=self.backbone, layers=layers, pre_trained=pre_trained)
        self.n_features_original, self.n_patches = _deduce_dims(self.feature_extractor, input_size, self.layers)

        n_features = n_features or _N_FEATURES_DEFAULTS.get(self.backbone)

        if n_features is None:
            raise ValueError(
                f"n_features must be specified for backbone {self.backbone}. "
                f"Default values are available for: {sorted(_N_FEATURES_DEFAULTS.keys())}"
            )

        assert (
            0 < n_features <= self.n_features_original
        ), f"for backbone {self.backbone}, 0 < n_features <= {self.n_features_original}, found {n_features}"

        self.n_features = n_features

        # pylint: disable=not-callable
        # Since idx is randomly selected, save it with model to get same results
        self.register_buffer(
            "idx",
            torch.tensor(sample(range(0, self.n_features_original), self.n_features)),
        )
        self.idx: Tensor
        self.loss = None
        self.anomaly_map_generator = AnomalyMapGenerator(image_size=input_size)

        self.gaussian = MultiVariateGaussian(self.n_features, self.n_patches)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Forward-pass image-batch (N, C, H, W) into model to extract features.

        Args:
            input_tensor: Image-batch (N, C, H, W)
            input_tensor: Tensor:

        Returns:
            Features from single/multiple layers.

        Example:
            >>> x = torch.randn(32, 3, 224, 224)
            >>> features = self.extract_features(input_tensor)
            >>> features.keys()
            dict_keys(['layer1', 'layer2', 'layer3'])

            >>> [v.shape for v in features.values()]
            [torch.Size([32, 64, 56, 56]),
            torch.Size([32, 128, 28, 28]),
            torch.Size([32, 256, 14, 14])]
        """

        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            embeddings = self.generate_embedding(features)

        if self.tiler:
            embeddings = self.tiler.untile(embeddings)

        if self.training:
            output = embeddings
        else:
            output = self.anomaly_map_generator(
                embedding=embeddings, mean=self.gaussian.mean, inv_covariance=self.gaussian.inv_covariance
            )
        return output

    def generate_embedding(self, features: dict[str, Tensor]) -> Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features (dict[str, Tensor]): Hierarchical feature map from a CNN (ResNet18 or WideResnet)

        Returns:
            Embedding vector
        """

        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        # subsample embeddings
        idx = self.idx.to(embeddings.device)
        embeddings = torch.index_select(embeddings, 1, idx)
        return embeddings
