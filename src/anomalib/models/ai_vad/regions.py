from __future__ import annotations

import torch
from torch import nn
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights


class RegionExtractor(nn.Module):
    def __init__(self, box_score_thresh: float = 0.8) -> None:
        super().__init__()

        weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        self.backbone = keypointrcnn_resnet50_fpn(weights=weights, box_score_thresh=box_score_thresh)

    def forward(self, batch):
        with torch.no_grad():
            regions = self.backbone(batch)

        return regions
