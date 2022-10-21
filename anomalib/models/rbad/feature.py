#  COSMONiO © All rights reserved.
#  This file is subject to the terms and conditions defined in file 'LICENSE.txt',
#  which is part of this source code package.

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
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

        # Load head module
        checkpoint = load_url(
            "https://files.cosmonio.com/combined_head_1_2-a9f83242.pth", check_hash=True, map_location=self.device
        )
        head_module.load_state_dict(checkpoint["module_state"])

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
        checkpoint = load_url(
            "https://files.cosmonio.com/rcnn_1_2-31296d99.pth", check_hash=True, map_location=self.device
        )
        self.load_state_dict(checkpoint["module_state"])
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
        self.head_module = BaseModel().get_head_module()
        self.rcnn_module = RCNN().load()
        self.normalizer = Normalizer()

    @torch.no_grad()
    def forward(self, image: np.ndarray, boxes: np.ndarray):

        if image.dtype in [np.float32, np.float64]:
            assert image.min() >= 0
            assert image.max() <= 1.0
            image = image * 255.0
            image = image.astype(np.uint8)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        assert image.dtype == np.uint8
        assert image.ndim == 3
        assert image.shape[2] == 3

        assert boxes.dtype == np.float32
        assert boxes.ndim == 2
        assert boxes.shape[1] == 4
        assert boxes.shape[0] > 0

        image, scale = self.transform(image)
        boxes_pt = torch.tensor(boxes)
        rois = torch.cat((torch.zeros(boxes_pt.size(0), 1), boxes_pt), 1).unsqueeze(0).to(self.device)
        rois *= scale

        base_feats = self.head_module(image)
        rcnn_feats = self.rcnn_module(base_feats, rois)

        return rcnn_feats.cpu().numpy()

    def transform(self, image: np.ndarray):
        # TODO: This will go the the dataloader in the future.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        in_shape = image.shape
        in_size_min = np.min(in_shape[0:2])
        in_size_max = np.max(in_shape[0:2])

        target_size = 600
        max_size = 1000

        scale = float(target_size) / float(in_size_min)
        if np.round(scale * in_size_max) > max_size:
            print("WARNING: cfg.MAX_SIZE exceeded. Using a different scaling ratio")
            scale = float(max_size) / float(in_size_max)

        image = cv2.resize(image, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image /= 255.0
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        image = image.permute(0, 3, 1, 2)
        image = self.normalizer.normalize(image)

        if torch.cuda.is_available():
            image = image.cuda()

        return image, scale
