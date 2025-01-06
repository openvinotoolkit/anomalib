"""PyTorch model for the SuperSimpleNet model implementation.

See Also:
    :class:`anomalib.models.image.supersimplenet.lightning_model.Supersimplenet`:
        SuperSimpleNet Lightning model.
"""

# Original Code
# Copyright (c) 2024 Blaž Rolih
# https://github.com/blaz-r/SuperSimpleNet.
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.nn import Parameter

from anomalib.data import InferenceBatch
from anomalib.models.components import GaussianBlur2d, TorchFXFeatureExtractor
from anomalib.models.image.supersimplenet.anomaly_generator import AnomalyGenerator


class SupersimplenetModel(nn.Module):
    """SuperSimpleNet Pytorch model.

    It consists of feature extractor, feature adaptor, anomaly generation mechanism and segmentation-detection module.

    Args:
        perlin_threshold (float): threshold value for Perlin noise thresholding during anomaly generation.
        backbone (str): backbone name
        layers (list[str]): backbone layers utilised
        stop_grad (bool): whether to stop gradient from class. to seg. head.
    """

    def __init__(
        self,
        perlin_threshold: float = 0.2,
        backbone: str = "wide_resnet50_2",
        layers: list[str] = ["layer2", "layer3"],  # noqa: B006
        stop_grad: bool = True,
    ) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor(backbone=backbone, layers=layers)

        channels = self.feature_extractor.get_channels_dim()
        self.adaptor = FeatureAdapter(channels)
        self.segdec = SegmentationDetectionModule(channel_dim=channels, stop_grad=stop_grad)
        self.anomaly_generator = AnomalyGenerator(noise_mean=0, noise_std=0.015, threshold=perlin_threshold)

        self.anomaly_map_generator = AnomalyMapGenerator(sigma=4)

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> InferenceBatch | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """SuperSimpleNet forward pass.

        Extract and process features, adapt them, generate anomalies (train only) and predict anomaly map and score.

        Args:
            images (torch.Tensor): Input images.
            masks (torch.Tensor): GT masks.
            labels (torch.Tensor): GT labels.

        Returns:
            inference: anomaly map and score
            training: anomaly map, score and GT masks and labels
        """
        output_size = images.shape[-2:]

        features = self.feature_extractor(images)
        adapted = self.adaptor(features)

        if self.training:
            masks = self.downsample_mask(masks, *features.shape[-2:])
            # make linter happy :)
            if labels is not None:
                labels = labels.type(torch.float32)

            features, masks, labels = self.anomaly_generator(
                adapted,
                masks,
                labels,
            )

            anomaly_map, anomaly_score = self.segdec(features)
            return anomaly_map, anomaly_score, masks, labels

        anomaly_map, anomaly_score = self.segdec(adapted)
        anomaly_map = self.anomaly_map_generator(anomaly_map, final_size=output_size)

        return InferenceBatch(anomaly_map=anomaly_map, pred_score=anomaly_score)

    @staticmethod
    def downsample_mask(masks: torch.Tensor, feat_h: int, feat_w: int) -> torch.Tensor:
        """Downsample the masks according to the feature dimensions.

        Primarily used in supervised setting.

        Args:
            masks (torch.Tensor): input GT masks
            feat_h (int): feature height.
            feat_w (int): feature width.

        Returns:
            (torch.Tensor): downsampled masks.
        """
        masks = masks.type(torch.float32)
        # best downsampling proposed by DestSeg
        masks = F.interpolate(
            masks.unsqueeze(1),
            size=(feat_h, feat_w),
            mode="bilinear",
        )
        return torch.where(
            masks < 0.5,
            torch.zeros_like(masks),
            torch.ones_like(masks),
        )


def init_weights(module: nn.Module) -> None:
    """Init weight of the model.

    Args:
        module (nn.Module): torch module.
    """
    if isinstance(module, nn.Linear | nn.Conv2d):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.BatchNorm1d | nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)


class FeatureExtractor(nn.Module):
    """Feature extractor module.

    Args:
        backbone (str): backbone name.
        layers (list[str]): list of layers used for extraction.
    """

    def __init__(self, backbone: str, layers: list[str], patch_size: int = 3) -> None:
        super().__init__()

        self.feature_extractor = TorchFXFeatureExtractor(
            backbone=backbone,
            return_nodes=layers,
            weights="IMAGENET1K_V1",
        )
        self.pooler = nn.AvgPool2d(
            kernel_size=patch_size,
            stride=1,
            padding=patch_size // 2,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Extract features from input tensor.

        Args:
            input_tensor: input tensor (images)

        Returns:
            (torch.Tensor): extracted feature map.
        """
        # extract features
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features = list(features.values())

        _, _, h, w = features[0].shape
        feature_map = []
        for layer in features:
            # upscale all to 2x the size of the first (largest)
            resized = F.interpolate(
                layer,
                size=(h * 2, w * 2),
                mode="bilinear",
            )
            feature_map.append(resized)
        # channel-wise concat
        feature_map = torch.cat(feature_map, dim=1)

        # neighboring patch aggregation
        return self.pooler(feature_map)

    def get_channels_dim(self) -> int:
        """Get feature channel dimension.

        Returns:
            (int): feature channel dimension.
        """
        # dryrun
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(torch.rand(1, 3, 256, 256))
        # sum channels
        return sum(feature.shape[1] for feature in features.values())


class FeatureAdapter(nn.Module):
    """Feature adapter used to adapt raw features for the task of anomaly detection.

    Args:
        channel_dim (int): channel dimension of features.
    """

    def __init__(self, channel_dim: int) -> None:
        super().__init__()
        # linear layer equivalent
        self.projection = nn.Conv2d(
            in_channels=channel_dim,
            out_channels=channel_dim,
            kernel_size=1,
            stride=1,
        )
        self.apply(init_weights)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Adapt features.

        Args:
            features (torch.Tensor): input features

        Returns:
            (torch.Tensor) adapted features
        """
        return self.projection(features)


class SegmentationDetectionModule(nn.Module):
    """SegmentationDetection module responsible for prediction of anomaly map and score.

    Args:
        channel_dim (int): channel dimension of features.
        stop_grad (bool): whether to stop gradient from class. head to seg. head.
    """

    def __init__(
        self,
        channel_dim: int,
        stop_grad: bool = False,
    ) -> None:
        super().__init__()
        self.stop_grad = stop_grad

        # 1x1 conv - linear layer equivalent
        self.seg_head = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=1024,
                kernel_size=1,
                stride=1,
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=1024,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

        # pooling for cls. conv out and map
        self.map_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.map_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.dec_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dec_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        # cls. head conv block
        self.cls_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_dim + 1,
                out_channels=128,
                kernel_size=5,
                padding="same",
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # cls. head fc block: 128 from dec and 2 from map, * 2 due to max and avg pool
        self.cls_fc = nn.Linear(in_features=128 * 2 + 2, out_features=1)

        self.apply(init_weights)

    def get_params(self) -> tuple[list[Parameter], list[Parameter]]:
        """Get segmentation and classification head parameters.

        Returns:
            seg. head parameters and class. head parameters.
        """
        seg_params = list(self.seg_head.parameters())
        dec_params = list(self.cls_conv.parameters()) + list(self.cls_fc.parameters())
        return seg_params, dec_params

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict anomaly map and anomaly score.

        Args:
            features: adapted features.

        Returns:
            predicted anomaly map and score.
        """
        # get anomaly map from seg head
        ano_map = self.seg_head(features)

        map_dec_copy = ano_map
        if self.stop_grad:
            map_dec_copy = map_dec_copy.detach()
        # dec conv layer takes feat + map
        mask_cat = torch.cat((features, map_dec_copy), dim=1)
        dec_out = self.cls_conv(mask_cat)

        # conv block result pooling
        dec_max = self.dec_max_pool(dec_out)
        dec_avg = self.dec_avg_pool(dec_out)

        # predicted map pooling (and stop grad)
        map_max = self.map_max_pool(ano_map)
        if self.stop_grad:
            map_max = map_max.detach()

        map_avg = self.map_avg_pool(ano_map)
        if self.stop_grad:
            map_avg = map_avg.detach()

        # final dec layer: conv channel max and avg and map max and avg
        dec_cat = torch.cat((dec_max, dec_avg, map_max, map_avg), dim=1).squeeze()
        ano_score = self.cls_fc(dec_cat).squeeze()

        return ano_map, ano_score


class AnomalyMapGenerator(nn.Module):
    """Final anomaly map generator, responsible for upscaling and smoothing.

    Args:
        sigma (float) Gaussian kernel sigma value.
    """

    def __init__(self, sigma: float) -> None:
        super().__init__()
        kernel_size = 2 * math.ceil(3 * sigma) + 1
        self.smoothing = GaussianBlur2d(kernel_size=kernel_size, sigma=4)

    def forward(self, out_map: torch.Tensor, final_size: tuple[int, int]) -> torch.Tensor:
        """Upscale and smooth anomaly map to get final anomaly map of same size as input image.

        Args:
            out_map (torch.Tensor): output anomaly map from seg. head.
            final_size (tuple[int, int]): size (h, w) of final anomaly map.

        Returns:
            torch.Tensor: final anomaly map.
        """
        # upscale & smooth
        anomaly_map = F.interpolate(out_map, size=final_size, mode="bilinear")
        return self.smoothing(anomaly_map)
