"""Callback to implement Selective Feature Modelling."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
from warnings import warn

import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor, tensor

from anomalib.models.components import SelectiveFeatureModel
from anomalib.utils.metrics import AdaptiveThreshold, metric_collection_from_names

logger = logging.getLogger(__name__)


class SelectiveFeatureModelCallback(Callback):
    """Callback which implements Selective Feature Model.

    Args:
        feature_percentage (float, optional): Percentage of features to store. Defaults to 0.1.
        save_dir (Path): Location where class features are saved to.
    """

    def __init__(self, save_dir: str, feature_percentage: float = 0.1) -> None:
        self.feature_percentage = feature_percentage
        self.feature_model = SelectiveFeatureModel(feature_percentage)
        self.max_features: Optional[Tensor] = None
        self.class_labels: List[str] = []
        self.save_dir = Path(save_dir)

        self.sub_pixel_threshold = AdaptiveThreshold().cpu()
        self.sub_pixel_metrics = metric_collection_from_names(["F1Score", "AUROC"], prefix="sfm_pixel_")
        self.sub_pixel_metrics.set_threshold(self.sub_pixel_threshold.value)

        self.output_masks: Optional[Tensor] = None
        self.selected_featuremaps: Optional[Dict[str, Tensor]] = None

    def _save_features(self, pl_module):
        # save category_features to disk

        with open(self.save_dir / "category_features.pkl", "wb") as pkl_file:
            pickle.dump(
                {
                    key: val.cpu().numpy()
                    for key, val in pl_module.model.anomaly_map_generator.category_features.items()
                },
                pkl_file,
            )

    def _load_features(self, pl_module):
        if hasattr(pl_module.model.anomaly_map_generator, "category_features"):
            # load category features
            with open(self.save_dir / "category_features.pkl", "rb") as pkl_file:
                data = pickle.load(pkl_file)
                pl_module.model.anomaly_map_generator.category_features = {
                    key: tensor(val, device=pl_module.device) for key, val in data.items()
                }

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

    def on_validation_epoch_end(self, _trainer: Trainer, pl_module: LightningModule) -> None:
        """Fit SFM model and reset max_features and class_labels."""

        if self.max_features is None:
            warn("`self.max_features is None")
            return

        logger.info("Fitting Selective Feature model")

        self.feature_model.fit(self.max_features, self.class_labels)

        for name, _ in self.feature_model.named_buffers():
            if name != "good":
                pl_module.model.anomaly_map_generator.category_features[name] = getattr(self.feature_model, name)[0]

        if hasattr(pl_module.model.anomaly_map_generator, "category_features"):
            self._save_features(pl_module)

        # reset arrays
        self.max_features = None
        self.class_labels = []

    def on_test_epoch_start(self, _trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset max_features and class_labels before testing starts."""
        self.max_features = None
        self.class_labels = []
        self.selected_featuremaps = None
        self.output_masks = None
        self._load_features(pl_module)

    def on_test_batch_end(
        self,
        _trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Tensor],
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Stores max activation values for all images.

        Args:
            pl_module (LightningModule): Anomaly module which contains the anomaly map generator.
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
                self.output_masks = outputs["mask"]
                self.selected_featuremaps = {
                    k: v.to(self.output_masks.device) for k, v in outputs["selected_features"].items()
                }
            else:
                self.max_features = torch.vstack([self.max_features, max_val_features])
                self.class_labels = np.hstack([self.class_labels, class_labels])
                self.output_masks = torch.vstack([self.output_masks, outputs["mask"]])
                self.selected_featuremaps = {
                    k: torch.vstack([self.selected_featuremaps[k], v.to(self.output_masks.device)])
                    for k, v in outputs["selected_features"].items()
                }

        # compute metrics
        # compute adaptive threshold
        selected_features = None
        anomaly_ids = []
        for idx, class_name in enumerate(outputs["class"]):
            if class_name == "good":
                continue
            anomaly_ids.append(idx)
            feature_map = pl_module.model.anomaly_map_generator.feature_to_anomaly_map(outputs["selected_features"][class_name][idx], feature=-1)
            if selected_features is None:
                selected_features = feature_map
            else:
                selected_features = torch.vstack([selected_features, feature_map])
        if selected_features is not None:
            self.sub_pixel_threshold.update(selected_features.flatten(), outputs["mask"][anomaly_ids].flatten().int())

    def on_test_epoch_end(self, _trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute sub-class testing accuracy."""

        self.sub_pixel_threshold.cpu()
        self.sub_pixel_metrics.cpu()
        self.sub_pixel_threshold.compute()
        self.sub_pixel_metrics.set_threshold(self.sub_pixel_threshold.value.item())

        if self.max_features is None:
            warn("`self.max_features is None")
            return

        if hasattr(pl_module.model.anomaly_map_generator, "category_features"):
            pl_module.model.anomaly_map_generator.category_features = dict()

        class_names = np.unique(self.class_labels)

        results: Dict[str, list] = {}
        for class_name in class_names:
            results[class_name] = []
        # sorted values and idx for entire feature set
        max_val, max_idx = torch.sort(self.max_features, descending=True)
        reduced_range = int(max_val.shape[1] * self.feature_percentage)
        # indexes of top self.feature_percentage features having max value
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

            # Get feature maps for predicted class
            feature_map = pl_module.model.anomaly_map_generator.feature_to_anomaly_map(self.selected_featuremaps[predicted_class][idx], feature=-1)
            # compute pixel metrics
            self.sub_pixel_metrics.update(feature_map.flatten(), self.output_masks[idx].flatten().int())

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

        pl_module.log_dict(self.sub_pixel_metrics.compute(), prog_bar=True)
