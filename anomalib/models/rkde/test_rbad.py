"""Region-based Anomaly Detection with Real Time Training and Analysis.

Test script to compare the anomalib implementation with the actual one.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data import InferenceDataset, get_datamodule
from anomalib.models import get_model
from anomalib.models.rkde.feature import FeatureExtractor as NousFeatureExtractor
from anomalib.models.rkde.model import NormalityModel
from anomalib.models.rkde.region import RegionExtractor as NousRegionExtractor
from anomalib.models.rkde.torch_model import RkdeModel
from anomalib.pre_processing import PreProcessor
from anomalib.pre_processing.pre_process import get_transforms
from anomalib.utils.callbacks import get_callbacks


# @pytest.mark.parametrize(
#     ["stage", "use_original"],
#     [("rpn", False), ("rcnn", False), ("rpn", True), ("rcnn", True)],
# )
# def test_output_shapes(stage, use_original):
def test_output_shapes() -> None:
    stage = "rcnn"
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
    # anomalib_rois, anomalib_features = torch_model.get_rois_and_features(data["image"].cuda())
    anomalib_rois = torch_model.region_extractor(data["image"].cuda())
    anomalib_features = torch_model.feature_extractor(data["image"].cuda(), anomalib_rois)

    assert nous_boxes[0].shape[0] == anomalib_rois.shape[0], "Number of boxes should be the same."
    assert np.allclose(nous_boxes, anomalib_rois[0].cpu().numpy(), atol=1e-02), "Boxes should be the same."
    assert nous_features.shape == anomalib_features.shape, "Feature shapes do not match."
    assert np.allclose(nous_features, anomalib_features.cpu().numpy(), atol=1e-02), "Features do not match."


def test_normality_model():
    seed_everything(42)

    config = get_configurable_parameters(model_name="rkde")
    datamodule = get_datamodule(config)
    model = get_model(config)
    callbacks = get_callbacks(config)

    # train new model
    config.trainer.limit_val_batches = 0
    trainer = Trainer(**config.trainer, callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)
    model.model.fit(model.embeddings)

    # fit old normality model
    nous_normality_model = NormalityModel()
    for feature_stack in model.embeddings:
        nous_normality_model.stage_features(feature_stack.cpu())
    nous_normality_model.commit()

    # infer image
    model.cuda()
    batch = next(iter(datamodule.test_dataloader()))
    features = model.model(batch["image"].cuda())
    nous_pred = nous_normality_model.evaluate(features.cpu(), as_density=True, ln=True)

    # infer image new
    model.cuda()
    model.model.eval()
    rois = model.model.region_extractor(batch["image"].cuda())
    features = model.model.feature_extractor(batch["image"].cuda(), rois)
    anomalib_pred = model.model.compute_kde_scores(features, as_log_likelihood=True)

    assert np.allclose(nous_pred, anomalib_pred.cpu(), atol=1e-02)


if __name__ == "__main__":
    test_output_shapes()
    test_normality_model()
