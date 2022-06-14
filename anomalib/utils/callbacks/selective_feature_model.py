import logging
from typing import Any, List, Optional

import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from anomalib.models.components import SelectiveFeatureModel

logger = logging.getLogger(__name__)


class SelectiveFeatureModelCallback(Callback):
    def __init__(self, feature_percentage: float = 0.1) -> None:
        self.feature_model = SelectiveFeatureModel(feature_percentage)
        self.max_features: Optional[Tensor] = None
        self.class_labels: List[str] = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        max_val_features = torch.vstack([x for x in outputs["max_activation_val"]])
        class_labels = np.hstack([x for x in outputs["class"]])

        if self.max_features is None:
            self.max_features = max_val_features
            self.class_labels = class_labels
        else:
            self.max_features = torch.vstack([self.max_features, max_val_features])
            self.class_labels = np.hstack([self.class_labels, class_labels])

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logger.info("Fitting Selective Feature model")

        self.feature_model.fit(self.max_features, self.class_labels)

        # reset arrays
        self.max_features = None
        self.class_labels = []

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.max_features = None
        self.class_labels = []

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        max_val_features = torch.vstack([x for x in outputs["max_activation_val"]])
        class_labels = np.hstack([x for x in outputs["class"]])

        if self.max_features is None:
            self.max_features = max_val_features
            self.class_labels = class_labels
        else:
            self.max_features = torch.vstack([self.max_features, max_val_features])
            self.class_labels = np.hstack([self.class_labels, class_labels])

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        class_names = np.unique(self.class_labels)
        # print(class_labels)
        # print(max_activation_val.shape)

        results = {}
        for class_name in class_names:
            results[class_name] = []
        # sorted values and idx for entire feature set
        max_val, max_idx = torch.sort(self.max_features, descending=True)
        reduced_range = int(max_val.shape[1] * 0.10)
        # indexes of top 10% FEATURES HAVING MAX VALUE
        top_max_idx = max_idx[:, 0:reduced_range]
        correct_class = []
        for idx, feature in enumerate(top_max_idx):
            scores = []
            for class_name in class_names:
                # print(class_name)
                stats = getattr(self.feature_model, class_name).cpu()
                score = stats[1][np.isin(stats[0], feature)].sum()
                scores.append(score)
            scores[np.where(class_names == "good")[0][0]] = 0
            if "combined" in class_names:
                scores[np.where(class_names == "combined")[0][0]] = 0
            if "thread" in class_names:
                scores[np.where(class_names == "thread")[0][0]] = 0
            predicted_class = class_names[scores.index(max(scores))]
            if self.class_labels[idx] not in ["good", "thread", "combined"]:
                # print(f"predicted: {predicted_class}, actual: {class_labels[idx]}")
                if predicted_class == self.class_labels[idx]:
                    correct_class.append(1)
                    results[self.class_labels[idx]].append(1)
                else:
                    correct_class.append(0)
                    results[self.class_labels[idx]].append(0)
        print("*********************")
        print(class_names)
        for class_name in class_names:
            if class_name not in ["good", "thread", "combined"]:
                print(
                    f"{class_name} accuracy ({sum(results[class_name])}/{len(results[class_name])}):"
                    f" {sum(results[class_name])/len(results[class_name])}"
                )
        print(f"average accuracy: {sum(correct_class)/len(correct_class)}")
