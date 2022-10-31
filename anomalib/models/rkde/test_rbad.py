"""Region-based Anomaly Detection with Real Time Training and Analysis.

Test script to compare the anomalib implementation with the actual one.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import albumentations as A
import cv2
import pytest
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from anomalib.data import InferenceDataset
from anomalib.models.rkde.feature import BaseModel as BaseModel1
from anomalib.models.rkde.feature_extractor import BaseModel as BaseModel2
from anomalib.models.rkde.feature_extractor import RegionExtractor as RegionExtractor2
from anomalib.models.rkde.region import RegionExtractor as RegionExtractor1
from anomalib.pre_processing import PreProcessor
from anomalib.pre_processing.pre_process import get_transforms


def test_base_model():
    tensor = torch.rand(1, 3, 224, 224)
    base_model1 = BaseModel1().get_head_module()
    base_model2 = BaseModel2().get_backbone()

    features1 = base_model1(tensor)
    features2 = base_model2(tensor)

    assert torch.allclose(features1, features2)


# @pytest.mark.parametrize(
#     ["stage", "use_original"],
#     [("rpn", False), ("rcnn", False), ("rpn", True), ("rcnn", True)],
# )
# def test_output_shapes(stage, use_original):
#     filename = "150.tif"
#     image = cv2.imread(filename)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Transformations.
#     transforms = get_transforms(config=A.Compose([A.Normalize(mean=0.0, std=1.0), ToTensorV2()]))
#     pre_process = PreProcessor(config=transforms)

#     # Get the data via dataloader
#     dataset = InferenceDataset(path=filename, pre_process=pre_process)
#     dataloader = DataLoader(dataset)
#     i, data = next(enumerate(dataloader))

#     # Create the region extractor.
#     # stage = "rpn"
#     # use_original = False
#     region_extractor1 = RegionExtractor1(stage=stage, use_original=use_original).eval().cuda()
#     region_extractor2 = RegionExtractor2(stage=stage, use_original=use_original).eval().cuda()

#     # Forward-Pass the input
#     boxes1 = region_extractor1([image])

#     # images = [image.to(device) for image in data["image"]]
#     # out2 = region_extractor2(images)
#     boxes2 = region_extractor2(data["image"].cuda())

#     assert boxes1[0].shape == boxes2[0].cpu().numpy().shape
