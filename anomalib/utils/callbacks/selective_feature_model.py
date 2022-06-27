"""Callback to implement Selective Feature Modelling."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Dict, List, Optional
from warnings import warn

import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor

from anomalib.models.components import SelectiveFeatureModel

logger = logging.getLogger(__name__)


class SelectiveFeatureModelCallback(Callback):
    """Callback which implements Selective Feature Model.

    Args:
        feature_percentage (float, optional): Percentage of features to store. Defaults to 0.1.
    """

    def __init__(self, feature_percentage: float = 0.1) -> None:
        self.feature_model = SelectiveFeatureModel(feature_percentage)
        self.max_features: Optional[Tensor] = None
        self.class_labels: List[str] = []

    def on_validation_batch_end(
        self,
        _trainer: Trainer,
        _pl_module: LightningModule,
        outputs: Dict[str, Tensor],
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Stores max activation values for all images.

        Args:
            outputs (Dict[str, Tensor]): Outputs containing `max_activation_val`.
        """
        if "max_activation_val" not in outputs.keys() and "class" not in outputs.keys():
            warn("Need both max_activation_val and class keys in outputs. Skipping SFM validation step")
        else:
            max_val_features = torch.vstack(list(outputs["max_activation_val"]))
            class_labels = np.hstack(list(outputs["class"]))

            if self.max_features is None:
                self.max_features = max_val_features
                self.class_labels = class_labels
            else:
                self.max_features = torch.vstack([self.max_features, max_val_features])
                self.class_labels = np.hstack([self.class_labels, class_labels])

    def on_validation_epoch_end(self, _trainer: Trainer, _pl_module: LightningModule) -> None:
        """Fit SFM model and reset max_features and class_labels."""

        if self.max_features is None:
            warn("`self.max_features is None")
            return

        logger.info("Fitting Selective Feature model")

        self.feature_model.fit(self.max_features, self.class_labels)

        # reset arrays
        self.max_features = None
        self.class_labels = []

    def on_test_epoch_start(self, _trainer: Trainer, _pl_module: LightningModule) -> None:
        """Reset max_features and class_labels before testing starts."""
        self.max_features = None
        self.class_labels = []

    def on_test_batch_end(
        self,
        _trainer: Trainer,
        _pl_module: LightningModule,
        outputs: Dict[str, Tensor],
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Stores max activation values for all images.

        Args:
            outputs (Dict[str, Tensor]): Outputs containing `max_activation_val`.
        """
        if "max_activation_val" not in outputs.keys() and "class" not in outputs.keys():
            warn("Need both max_activation_val and class keys in outputs. Skipping SFM test step")
        else:
            max_val_features = torch.vstack(list(outputs["max_activation_val"]))
            class_labels = np.hstack(list(outputs["class"]))

            if self.max_features is None:
                self.max_features = max_val_features
                self.class_labels = class_labels
            else:
                self.max_features = torch.vstack([self.max_features, max_val_features])
                self.class_labels = np.hstack([self.class_labels, class_labels])

    def on_test_epoch_end(self, _trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute sub-class testing accuracy."""
        if self.max_features is None:
            warn("`self.max_features is None")
            return

        class_names = np.unique(self.class_labels)

        results: Dict[str, list] = {}
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
                if predicted_class == self.class_labels[idx]:
                    correct_class.append(1)
                    results[self.class_labels[idx]].append(1)
                else:
                    correct_class.append(0)
                    results[self.class_labels[idx]].append(0)
        for class_name in class_names:
            if class_name not in ["good", "thread", "combined"]:
                pl_module.log(
                    f"{class_name} accuracy",
                    sum(results[class_name]) / len(results[class_name]),
                )
        pl_module.log("Average sub-class accuracy", sum(correct_class) / len(correct_class))
