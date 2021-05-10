import os
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from attrdict import AttrDict
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from sklearn.metrics import roc_auc_score

from anomalib.models.anocls.features import resnet50_feature_extractor
from anomalib.models.anocls.normality_model import NormalityModel


class Callbacks:
    def __init__(self, args):
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


class AnoCLSModel(pl.LightningModule):
    def __init__(self, hparams: AttrDict):
        super().__init__()
        self.hparams = hparams
        self.threshold_steepness = 0.05
        self.threshold_offset = 12

        self.feature_extractor = resnet50_feature_extractor().eval()

        self.normality_model = NormalityModel(
            filter_count=hparams.max_training_points,
            threshold_steepness=self.threshold_steepness,
            threshold_offset=self.threshold_offset,
        )
        self.callbacks = Callbacks(hparams)()

    def configure_optimizers(self):
        return None

    def training_step(self, batch, batch_idx):
        self.feature_extractor.eval()
        feature_vector = self.feature_extractor(batch["image"]).detach()
        return {"feature_vector": feature_vector}

    def training_epoch_end(self, outputs):
        feature_stack = torch.vstack([output["feature_vector"] for output in outputs])
        self.normality_model.fit(feature_stack)

    def validation_step(self, batch, batch_idx):
        self.feature_extractor.eval()
        images, mask = batch["image"], batch["mask"]
        feature_vector = self.feature_extractor(images).detach()
        probability = self.normality_model.predict(feature_vector)
        prediction = 1 if probability > self.hparams.confidence_threshold else 0
        ground_truth = int(np.any(mask.cpu().numpy()))
        return {"probability": probability, "prediction": prediction, "ground_truth": ground_truth}

    def validation_epoch_end(self, outputs):
        pred = [output["probability"] for output in outputs]
        gt = [int(output["ground_truth"]) for output in outputs]
        auc = roc_auc_score(np.array(gt), np.array(torch.hstack(pred)))
        self.log(name="auc", value=auc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)
