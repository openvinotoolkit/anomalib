import os
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from attrdict import AttrDict
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet50

from anomalib.models.dfkde.normality_model import NormalityModel
from anomalib.models.shared.feature_extractor import FeatureExtractor


class Callbacks:
    def __init__(self, args: DictConfig):
        self.args = args

    def get_callbacks(self) -> List[Callback]:
        checkpoint = ModelCheckpoint(
            dirpath=os.path.join(self.args.project_path, "weights"),
            filename="model",
            monitor=self.args.metric,
        )
        callbacks = [checkpoint]
        return callbacks

    def __call__(self):
        return self.get_callbacks()


class DFKDEModel(pl.LightningModule):
    def __init__(self, hparams: AttrDict):
        super().__init__()
        self.hparams = hparams
        self.threshold_steepness = 0.05
        self.threshold_offset = 12

        self.feature_extractor = FeatureExtractor(backbone=resnet50(pretrained=True), layers=["avgpool"]).eval()

        self.normality_model = NormalityModel(
            filter_count=hparams.max_training_points,
            threshold_steepness=self.threshold_steepness,
            threshold_offset=self.threshold_offset,
        )
        self.callbacks = Callbacks(hparams)()

    def configure_optimizers(self):
        return None

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        self.feature_extractor.eval()
        layer_outputs = self.feature_extractor(batch["image"])
        feature_vector = torch.hstack(list(layer_outputs.values())).detach().squeeze()
        return {"feature_vector": feature_vector}

    def training_epoch_end(self, outputs: dict):
        feature_stack = torch.vstack([output["feature_vector"] for output in outputs])
        self.normality_model.fit(feature_stack)

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        self.feature_extractor.eval()
        images, mask = batch["image"], batch["mask"]
        layer_outputs = self.feature_extractor(batch["image"])
        feature_vector = torch.hstack(list(layer_outputs.values())).detach()
        probability = self.normality_model.predict(feature_vector.view(feature_vector.shape[:2]))
        prediction = 1 if probability > self.hparams.confidence_threshold else 0
        ground_truth = int(np.any(mask.cpu().numpy()))
        return {"probability": probability, "prediction": prediction, "ground_truth": ground_truth}

    def validation_epoch_end(self, outputs: dict):
        pred = [output["probability"] for output in outputs]
        gt = [int(output["ground_truth"]) for output in outputs]
        auc = roc_auc_score(np.array(gt), np.array(torch.hstack(pred)))
        self.log(name="auc", value=auc, on_epoch=True, prog_bar=True)

    def test_step(self, batch: dict, batch_idx: dict) -> dict:
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs: dict):
        self.validation_epoch_end(outputs)
