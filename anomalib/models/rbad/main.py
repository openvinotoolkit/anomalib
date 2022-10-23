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
from anomalib.models.rbad.region import RegionExtractor as RegionExtractor1
from anomalib.models.rbad.region_extractor import RegionExtractor as RegionExtractor2
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
    region_extractor2 = RegionExtractor2(stage=stage, use_original=use_original).eval().cuda()

    # Forward-Pass the input
    out2 = region_extractor2(data["image"].cuda())


if __name__ == "__main__":
    main()
