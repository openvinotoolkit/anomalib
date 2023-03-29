from __future__ import annotations

import torch
from torch import nn
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights


class RegionExtractor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        self.backbone = keypointrcnn_resnet50_fpn(weights=weights)

    def forward(self, batch):
        with torch.no_grad():
            regions = self.backbone(batch)

        return regions
