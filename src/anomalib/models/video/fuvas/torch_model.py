"""PyTorch model implementation of Fuvas.

This module provides a PyTorch implementation of the Fuvas model for anomaly
detection. The model extracts deep features from video clips using a pre-trained 3D CNN/transformer
backbone and fits a low-rank factorization model on these features to detect anomalies.

Example:
    >>> import torch
    >>> from anomalib.models.video.fuvas.torch_model import FUVASModel
    >>> model = FUVASModel(
    ...     backbone="swin3d_b",
    ...     layer="features.6.1",
    ... )
    >>> batch = torch.randn(3, 3, 8, 224, 224)
    >>> features = model(batch)  # Returns features during training
    >>> predictions = model(batch)  # Returns scores during inference

Notes:
    The model uses a pre-trained backbone to extract features and fits a PCA
    transformation followed by a low-rank factorization model during training. No gradient
    updates are performed on the backbone.
"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from skimage.transform import resize
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torchvision.models.video import Swin3D_B_Weights, swin3d_b
from torchvision.models.feature_extraction import create_feature_extractor

from anomalib.data import InferenceBatch

# from .feature_extractor import FeatureExtractor


class FUVASModel(nn.Module):
    """Fuvas model for video anomaly segmentation.

    The model extracts deep features from video clips using a pre-trained 3D CNN/transformer backbone
    and fits a low-rank factorization model on these features to detect anomalies.

    Args:
        backbone (str): Pre-trained model backbone from torchvision.
        layer (str): Layer from which to extract features.
        spatial_pool (bool, optional): Whether to pool features spatially.
            Defaults to ``True``.
        pre_trained (bool, optional): Whether to use pre-trained backbone.
            Defaults to ``True``.
        pooling_kernel_size (int, optional): Kernel size to pool features.
            Defaults to ``4``.
        n_comps (float, optional): Ratio for PCA components calculation.
            Defaults to ``0.97``.
        do_seg (bool, optional): Whether to do video anomaly segmentation or detection
            Default to ``False``.

    Example:
        >>> model = FUVASModel(
        ...     backbone="swin3d_b",
        ...     layer="features.6.1",
        ... )
        >>> input_tensor = torch.randn(batch, clip_len, 3, 448(256), 512(256))
        >>> output = model(input_tensor)
    """

    def __init__(
        self,
        backbone: str,
        layer: str,
        spatial_pool: bool,
        pre_trained: bool = True,
        pooling_kernel_size: int = 4,
        n_comps: float = 0.98,
        do_seg: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling_kernel_size = pooling_kernel_size
        self.n_components = n_comps
        self.pca_model = PCA(n_components=self.n_components)
        self.do_seg = do_seg
        self.layer = layer
        self.spatial_pool = spatial_pool
        if backbone in {"i3d_r50", "x3d_l", "x3d_xs", "x3d_s", "x3d_m"}:
            # Load PyTorchVideo model from torch hub
            # Supported models: i3d_r50, x3d_l, x3d_xs, x3d_s, x3d_m
            # Common input sizes: (448,512)
            # Common feature extraction layers:
            #   - blocks.6.dropout: 2048 dimensional features
            #   - blocks.5.res_blocks.2: Features before final layers
            net = torch.hub.load("facebookresearch/pytorchvideo", model=backbone, pretrained=pre_trained)

        elif backbone == "swin3d_b":
            # Load Swin3D-B model pre-trained on Kinetics-400 and ImageNet-22K
            # Alternative weights: KINETICS400_V1 (pre-trained on Kinetics-400 only)
            # Input normalization:
            #   - KINETICS400_IMAGENET22K_V1: Uses ImageNet normalization
            #   - KINETICS400_V1: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # Common input size: (256,256), minimum temporal dimension: 1
            net = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1)
        else:
            msg = (
                f"Backbone '{backbone}' is not supported. "
                f"Use one of: 'i3d_r50', 'x3d_l', 'x3d_xs', 'x3d_s', 'x3d_m', 'swin3d_b'."
            )
            raise ValueError(msg)

        net.eval()
        self.feature_extractor = create_feature_extractor(net, return_nodes=[layer])

        self.peak = 0

    def fit(self, dataset: torch.Tensor) -> None:
        """Fit PCA model to dataset.

        Args:
            dataset (torch.Tensor): Input dataset with shape
                ``(n_samples, n_features)``.
        """
        self.pca_model.fit(dataset.numpy())
        oi_j = self.pca_model.transform(dataset.numpy())
        oi_recon = self.pca_model.inverse_transform(oi_j)
        self.peak = abs(dataset.numpy() - oi_recon)

    def score(self, features: torch.Tensor, feature_shapes: tuple) -> torch.Tensor:
        """Compute anomaly scores.

        Scores are PCA-based feature reconstruction error (FRE) scores.

        Args:
            features (torch.Tensor): Features for scoring with shape
                ``(n_samples, n_features)``.
            feature_shapes (tuple): Shape of features tensor for anomaly map.

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: Tuple containing
                (scores, anomaly_maps).
        """
        feats_projected = self.pca_model.transform(features.numpy())
        feats_reconstructed = self.pca_model.inverse_transform(feats_projected)
        fre_prereshape = np.square(features.numpy() - feats_reconstructed)
        fre_prereshape_np = fre_prereshape
        fre = fre_prereshape_np.reshape(feature_shapes)
        score_map_np = np.sum(fre, axis=(1, 2))  # NxTxCxHxW->NxHxW
        score_map = torch.from_numpy(score_map_np)
        score = torch.from_numpy(np.sum(score_map_np, axis=(1, 2)))  # NxHxW->N

        return score if not self.do_seg else (score, score_map)

    def get_features(self, batch: torch.Tensor) -> torch.Tensor:
        """Extract features from the pretrained network.

        Args:
            batch (torch.Tensor): Input video clips with shape
                ``(batch_size, num_clips, channels, height, width)``.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Size]]: Features during
                training, or tuple of (features, feature_shapes) during inference.
        """
        with torch.no_grad():
            self.feature_extractor.eval()
            batch = torch.permute(batch, (0, 2, 1, 3, 4))
            out_dict = self.feature_extractor(batch)
            if self.layer.startswith("features"):
                out = torch.permute(out_dict[self.layer], (0, 1, 4, 2, 3))
            else:
                out = out_dict[self.layer]
            # pool
            if self.spatial_pool:
                if len(out.shape) == 5:
                    pool_features = F.avg_pool3d(out, (1, self.pooling_kernel_size, self.pooling_kernel_size))
                elif len(out.shape) == 4:
                    pool_features = F.avg_pool2d(out, (self.pooling_kernel_size, self.pooling_kernel_size))
                else:
                    pool_features = F.avg_pool1d(out, self.pooling_kernel_size)
                feature_shape = pool_features.shape
            else:
                feature_shape = list(out.shape)
                fea_vector = out.reshape(feature_shape[0], -1)
                pool_features = F.avg_pool1d(fea_vector, self.pooling_kernel_size)
                feature_shape[1] = feature_shape[1] // self.pooling_kernel_size

        features = pool_features.cpu().reshape(feature_shape[0], -1)
        return features, feature_shape

    def forward(self, batch: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Compute anomaly predictions from input images.

        Args:
            batch (torch.Tensor): Input images with shape
                ``(batch_size, clip_len,channels, height, width)``.

        Returns:
            Union[torch.Tensor, InferenceBatch]: Model predictions. During
                training returns features tensor. During inference returns
                ``InferenceBatch`` with prediction scores and anomaly maps.
        """
        feature_vector, feature_shapes = self.get_features(batch)
        anomaly_map = None
        if self.do_seg:
            pred_score, anomaly_map = self.score(feature_vector, feature_shapes)
        else:
            pred_score = self.score(feature_vector, feature_shapes)
        if anomaly_map is not None:
            anomaly_map_stack = []
            for map_one in anomaly_map:
                map_one_np = map_one.numpy()
                anomaly_map_stack.append(resize(map_one_np, tuple(batch.shape[-2:])))
            anomaly_map_stack = np.stack(anomaly_map_stack)
            anomaly_map_stack = torch.from_numpy(anomaly_map_stack)
            anomaly_map_stack = torch.unsqueeze(anomaly_map_stack, 1).to(batch.device)
        pred_score = pred_score.to(batch.device)
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map_stack)
