from __future__ import annotations

import torch
from torch import nn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2


class RegionExtractor(nn.Module):
    def __init__(self, box_score_thresh: float = 0.8) -> None:
        super().__init__()

        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.backbone = maskrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=box_score_thresh, rpn_nms_thresh=0.3)

    def forward(self, batch):
        with torch.no_grad():
            regions = self.backbone(batch)

        return regions
