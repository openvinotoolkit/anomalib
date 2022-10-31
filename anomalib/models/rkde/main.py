"""Region-based Anomaly Detection with Real Time Training and Analysis.

Test script to compare the anomalib implementation with the actual one.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from anomalib.data import InferenceDataset
from anomalib.models.rkde.feature import FeatureExtractor as FeatureExtractor1
from anomalib.models.rkde.feature_extractor import RegionExtractor as RegionExtractor2
from anomalib.models.rkde.region import RegionExtractor as RegionExtractor1
from anomalib.pre_processing import PreProcessor
from anomalib.pre_processing.pre_process import get_transforms


def main():
    filename = "150.tif"
    image = cv2.imread(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformations.
    transforms = get_transforms(config=A.Compose([A.Normalize(mean=0.0, std=1.0), ToTensorV2()]))
    pre_process = PreProcessor(config=transforms)

    # Get the data via dataloader
    dataset = InferenceDataset(path=filename, pre_process=pre_process)
    dataloader = DataLoader(dataset)
    i, data = next(enumerate(dataloader))

    # Create the region extractor.
    stage = "rcnn"
    use_original = False
    region_extractor1 = RegionExtractor1(stage=stage, use_original=use_original).eval().cuda()
    region_extractor2 = RegionExtractor2(stage=stage, use_original=use_original).eval().cuda()

    # Forward-Pass the input
    boxes1 = region_extractor1([image])
    boxes2 = region_extractor2(data["image"].cuda())

    feature_extractor1 = FeatureExtractor1().eval().cuda()
    features1 = feature_extractor1(image, boxes1)


if __name__ == "__main__":
    main()
