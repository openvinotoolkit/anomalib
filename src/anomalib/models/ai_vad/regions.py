from __future__ import annotations

import torch
from torch import nn, Tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import pickle
from pathlib import Path


class RegionExtractor(nn.Module):
    def __init__(self, box_score_thresh: float = 0.8) -> None:
        super().__init__()

        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.backbone = maskrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=box_score_thresh, rpn_nms_thresh=0.3)

    def forward(self, batch):
        with torch.no_grad():
            regions = self.backbone(batch)

        return regions


class RegionExtractorOrig(nn.Module):
    def __init__(self, box_score_thresh: float = 0.8) -> None:
        super().__init__()

        self.regions_path_train = "/home/djameln/Accurate-Interpretable-VAD/data/ped2/ped2_bboxes_dict_train.pickle"
        self.regions_path_test = "/home/djameln/Accurate-Interpretable-VAD/data/ped2/ped2_bboxes_dict_test.pickle"

        with open(self.regions_path_train, "rb") as f:
            self.regions_train = pickle.load(f)
        with open(self.regions_path_test, "rb") as f:
            self.regions_test = pickle.load(f)

    def forward(self, batch, video_paths, frames, train=True):
        outputs = []
        for video_path, frame in zip(video_paths, frames):
            vid_id = Path(video_path).name
            frame_id = frame[0].int().item()
            if train:
                bboxes = self.regions_train[vid_id][frame_id]
            else:
                bboxes = self.regions_test[vid_id][frame_id]
            outputs.append({"boxes": Tensor(bboxes).to(batch.device)})

        return outputs
