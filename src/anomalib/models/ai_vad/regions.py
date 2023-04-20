from __future__ import annotations

import torch
from torch import nn
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
import numpy as np

class RegionExtractor(nn.Module):
    def __init__(self, box_score_thresh: float = 0.8) -> None:
        super().__init__()

        weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        self.backbone = keypointrcnn_resnet50_fpn(weights=weights, box_score_thresh=box_score_thresh)

    def forward(self, batch):
        with torch.no_grad():
            regions = self.backbone(batch)

        return regions


class RegionExtractorOrig(nn.Module):
    def __init__(self, box_score_thresh: float = 0.8) -> None:
        super().__init__()

        self.regions_path_train = "/home/djameln/Accurate-Interpretable-VAD/data/ped2/ped2_bboxes_train.npy"
        self.regions_path_test = "/home/djameln/Accurate-Interpretable-VAD/data/ped2/ped2_bboxes_test.npy"

        self.regions_train = np.load(self.regions_path_train, allow_pickle=True)
        self.regions_test = np.load(self.regions_path_test, allow_pickle=True)

    def forward(self, video_path, frames, train=True):
        pass


