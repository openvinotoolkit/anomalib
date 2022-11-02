"""Region-based Anomaly Detection with Real Time Training and Analysis.

Test script to compare the anomalib implementation with the actual one.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from anomalib.data import InferenceDataset
from anomalib.models.rkde.feature import FeatureExtractor as NousFeatureExtractor
from anomalib.models.rkde.region import RegionExtractor as NousRegionExtractor
from anomalib.models.rkde.torch_model import RkdeModel
from anomalib.pre_processing import PreProcessor
from anomalib.pre_processing.pre_process import get_transforms


# @pytest.mark.parametrize(
#     ["stage", "use_original"],
#     [("rpn", False), ("rcnn", False), ("rpn", True), ("rcnn", True)],
# )
# def test_output_shapes(stage, use_original):
def test_output_shapes() -> None:
    stage="rcnn"
    use_original = False

    # NOUS implementation
    filename = "anomalib/models/rkde/150.tif"
    image = cv2.imread(filename)

    nous_region_extractor = NousRegionExtractor(stage=stage, use_original=use_original).eval().cuda()
    nous_feature_extractor = NousFeatureExtractor().eval().cuda()
    nous_boxes = nous_region_extractor([image])
    nous_features = nous_feature_extractor(image, nous_boxes[0])

    # Anomalib Implementation.
    # 1. Data
    transforms = get_transforms(config=A.Compose([A.Normalize(mean=0.0, std=1.0), ToTensorV2()]))
    pre_process = PreProcessor(config=transforms)
    dataset = InferenceDataset(path=filename, pre_process=pre_process)
    dataloader = DataLoader(dataset)
    i, data = next(enumerate(dataloader))

    # 2. Model
    torch_model = RkdeModel(region_extractor_stage=stage).eval().cuda()
    anomalib_rois, anomalib_features = torch_model.get_rois_and_features(data["image"].cuda())

    assert len(nous_boxes[0]) == len(anomalib_rois), "Number of boxes should be the same."
    assert np.allclose(nous_boxes, anomalib_rois.cpu().numpy()), "Boxes should be the same."
    assert nous_features.shape == anomalib_features.shape, "Feature shapes do not match."
    assert np.allclose(nous_features, anomalib_features.cpu().numpy(), atol=1e-02), "Features do not match."
