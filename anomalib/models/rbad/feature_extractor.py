"""Region-based Anomaly Detection with Real Time Training and Analysis.

Feature Extractor.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
from torch.utils.model_zoo import load_url
from torchvision.ops import RoIAlign
from utils.normalization import Normalizer


class BaseModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = models.alexnet(pretrained=pretrained)
        self.head_module = self.get_head_module()
        self.tail_module, self.tail_dimensions = self.get_tail_module()

    def get_head_module(self, module_type: str = "combined_head") -> nn.Sequential:
        head_module = self.model.features[:-1]
        head_module.module_type = module_type

        # # Load head module
        # checkpoint = load_url(
        #     "https://files.cosmonio.com/combined_head_1_2-a9f83242.pth", check_hash=True, map_location=self.device
        # )
        # head_module.load_state_dict(checkpoint["module_state"])

        return head_module

    def get_tail_module(self, module_type: str = "tail") -> Tuple[nn.Sequential, int]:
        # Get tail network
        tail_module = self.model.classifier[0:-1]
        tail_module.module_type = module_type

        # Get the dimension of the output features of the tail net.
        tail_dimensions = None
        for m in tail_module.modules():
            if isinstance(m, torch.nn.Linear):
                tail_dimensions = m.out_features

        return tail_module, tail_dimensions


class RCNN(nn.Module):
    def __init__(self):
        super(RCNN, self).__init__()
        self.module_type = "rcnn"
        self.branch_labels = get_branch_labels()
        self.n_classes, self.n_adjectives, self.n_verbs = [len(labels) for labels in self.branch_labels]
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        pooling_size = 6
        pooling_scale = 1.0 / 16.0

        # Create RoI Align Network.
        self.RoIAlign = RoIAlign((pooling_size, pooling_size), pooling_scale, 0)

        # Create R-CNN Tail Module.
        tail_module, tail_dimensions = BaseModel().get_tail_module()
        self.RCNN_tail = tail_module
        self.RCNN_cls_score = nn.Linear(tail_dimensions, self.n_classes)
        self.RCNN_adjective_score = nn.Linear(tail_dimensions, self.n_adjectives)
        self.RCNN_verb_score = nn.Linear(tail_dimensions, self.n_verbs)

    def load(self):
        # checkpoint = load_url(
        #     "https://files.cosmonio.com/rcnn_1_2-31296d99.pth", check_hash=True, map_location=self.device
        # )
        # self.load_state_dict(checkpoint["module_state"])
        return self

    def forward(self, base_feat, rois):
        # n_rois x 256 x 6 x 6 (AlexNet)
        pooled_feat = self.RoIAlign(base_feat, rois.view(-1, 5))

        # n_rois x 4096
        sem_feat = self.RCNN_tail(pooled_feat.view(pooled_feat.size(0), -1))
        return sem_feat


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.backbone = BaseModel().get_head_module()
        self.rcnn_module = RCNN().load()

    @torch.no_grad()
    def forward(self, images: Tensor, boxes: List[Tensor]) -> Tensor:
        # NOTE: This transform does not return the exact same output as cv2.
        images, scale = self.transform(images)

        # Process RoIs.
        # Convert list of boxes to batch format.
        rois = [box.unsqueeze(0) for box in boxes]
        rois = torch.cat(boxes, dim=0)
        # Add zero column for the scores.
        rois = F.pad(input=rois, pad=(1, 0, 0, 0), mode="constant", value=0)
        # Scale the RoIs based on the the new image size.
        rois *= scale

        # Extract RCNN features.
        backbone_features = self.backbone(images)
        rcnn_features = self.rcnn_module(backbone_features, rois)

        return rcnn_features

    def transform(self, input_tensor: Tensor) -> Tuple[Tensor, float]:
        # Apply the feature extractor transforms
        height, width = input_tensor.shape[2:]
        shorter_image_size, longer_image_size = min(height, width), max(height, width)
        target_image_size, max_image_size = 600, 1000

        scale = target_image_size / shorter_image_size
        if round(scale * longer_image_size) > max_image_size:
            print("WARNING: cfg.MAX_SIZE exceeded. Using a different scaling ratio")
            scale = max_image_size / longer_image_size

        # TODO: Use scale_factor
        # resized_image = F.interpolate(input_tensor, scale_factor=scale, mode="bilinear", align_corners=False)
        resized_tensor = F.interpolate(input_tensor, size=(600, 904), mode="bilinear", align_corners=False)

        # Apply the same transformation as the original model.
        mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(input_tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(input_tensor.device)
        normalized_tensor = (resized_tensor - mean) / std

        return normalized_tensor, scale


def get_branch_labels():
    demo_classes = [
        "__background__",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    demo_adjectives = [
        "old",
        "gold",
        "yellow",
        "bright",
        "pink",
        "bare",
        "parked",
        "tan",
        "open",
        "blonde",
        "blue",
        "little",
        "purple",
        "young",
        "long",
        "plastic",
        "black",
        "closed",
        "wet",
        "orange",
        "white",
        "red",
        "beige",
        "brown",
        "colourful",
        "big",
        "dark",
        "glass",
        "striped",
        "silver",
        "short",
        "wooden",
        "light",
        "clear",
        "metal",
        "grey",
        "cloudy",
        "large",
        "concrete",
        "green",
        "empty",
        "small",
        "tall",
        "brick",
        "round",
    ]
    demo_verbs = [
        "walking",
        "skiing",
        "talking",
        "hanging",
        "smiling",
        "skateboarding",
        "grazing",
        "shining",
        "standing",
        "flying",
        "looking",
        "waiting",
        "holding",
        "riding",
        "watching",
        "eating",
        "sitting",
        "running",
        "jumping",
        "moving",
        "surfing",
        "laying",
        "leaning",
        "growing",
        "playing",
    ]
    return demo_classes, demo_adjectives, demo_verbs
