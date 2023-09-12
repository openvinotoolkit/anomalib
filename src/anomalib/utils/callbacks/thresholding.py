"""Thresholding callback."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import Any

from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig, ListConfig
from torch import Tensor

from anomalib import trainer
from anomalib.models import AnomalyModule
from anomalib.utils.metrics.thresholding import BaseAnomalyThreshold, F1AdaptiveThreshold


class _ThresholdingCallback(Callback):
    """Setup/apply thresholding.

    Note: This callback is set within the AnomalibTrainer.
    """

    def __init__(
        self,
        threshold: BaseAnomalyThreshold
        | tuple[BaseAnomalyThreshold, BaseAnomalyThreshold]
        | DictConfig
        | ListConfig
        | str = F1AdaptiveThreshold(),
    ) -> None:
        super().__init__()
        self._initialize_thresholds(threshold)
        self.image_threshold: BaseAnomalyThreshold
        self.pixel_threshold: BaseAnomalyThreshold

    def setup(self, trainer: "trainer.AnomalibTrainer", pl_module: AnomalyModule, stage: str) -> None:
        if not hasattr(pl_module, "image_threshold"):
            pl_module.image_threshold = self.image_threshold
        if not hasattr(pl_module, "pixel_threshold"):
            pl_module.pixel_threshold = self.pixel_threshold

    def on_validation_epoch_start(self, trainer: "trainer.AnomalibTrainer", pl_module: AnomalyModule) -> None:
        self._reset(pl_module)

    def on_validation_batch_end(
        self,
        trainer: "trainer.AnomalibTrainer",
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs is not None:
            self._outputs_to_cpu(outputs)
            self._update(pl_module, outputs)

    def on_validation_epoch_end(self, trainer: "trainer.AnomalibTrainer", pl_module: AnomalyModule) -> None:
        self._compute(pl_module)

    def _setup(self, pl_module: AnomalyModule) -> None:
        pl_module.image_threshold = self.image_threshold
        pl_module.pixel_threshold = self.pixel_threshold

    def _update(self, pl_module: AnomalyModule, outputs: STEP_OUTPUT) -> None:
        pl_module.image_threshold.cpu()
        pl_module.image_threshold.update(outputs["pred_scores"], outputs["label"].int())
        if "mask" in outputs.keys() and "anomaly_maps" in outputs.keys():
            pl_module.pixel_threshold.cpu()
            pl_module.pixel_threshold.update(outputs["anomaly_maps"], outputs["mask"].int())

    def _reset(self, pl_module: AnomalyModule) -> None:
        pl_module.image_threshold.reset()
        pl_module.pixel_threshold.reset()

    def _compute(self, pl_module: AnomalyModule) -> None:
        pl_module.image_threshold.compute()
        if pl_module.pixel_threshold._update_called:
            pl_module.pixel_threshold.compute()
        else:
            pl_module.pixel_threshold.value = pl_module.image_threshold.value

    def _outputs_to_cpu(self, output: STEP_OUTPUT):
        if isinstance(output, dict):
            for key, value in output.items():
                output[key] = self._outputs_to_cpu(value)
        elif isinstance(output, Tensor):
            output = output.cpu()
        return output

    def _initialize_thresholds(
        self,
        threshold: BaseAnomalyThreshold
        | tuple[BaseAnomalyThreshold, BaseAnomalyThreshold]
        | DictConfig
        | ListConfig
        | str,
    ) -> None:
        if isinstance(threshold, BaseAnomalyThreshold):
            self.image_threshold = threshold
            self.pixel_threshold = threshold
        elif isinstance(threshold, tuple):
            self.image_threshold = threshold[0]
            self.pixel_threshold = threshold[1]
        elif isinstance(threshold, (str, DictConfig, ListConfig)):
            self._load_from_config(threshold)
        else:
            raise ValueError(f"Invalid threshold type {type(threshold)}")

    def _load_from_config(self, threshold: DictConfig | str | ListConfig) -> None:
        """Loads the thresholding class based on the config.

        Example:
            threshold: F1AdaptiveThreshold
            or
            threshold:
                class_path: F1AdaptiveThreshold
                init_args:
                    -
            or
            threshold:
                - adaptive
                - adaptive
            or
            threshold:
                - class_path: F1AdaptiveThreshold
                    init_args:
                        -
                - class_path: F1AdaptiveThreshold
        """
        if isinstance(threshold, (str, DictConfig)):
            self.image_threshold = self._get_threshold_from_config(threshold)
            self.pixel_threshold = self.image_threshold.clone()
        elif isinstance(threshold, ListConfig):
            self.image_threshold = self._get_threshold_from_config(threshold[0])
            self.pixel_threshold = self._get_threshold_from_config(threshold[1])
        else:
            raise ValueError(f"Invalid threshold config {threshold}")

    def _get_threshold_from_config(self, threshold: DictConfig | str) -> BaseAnomalyThreshold:
        if isinstance(threshold, str):
            threshold = DictConfig({"class_path": threshold})

        class_path = threshold.class_path
        init_args = threshold.init_args if "init_args" in threshold else {}

        if len(class_path.split(".")) == 1:
            module_path = "anomalib.utils.metrics.thresholding"

        else:
            module_path = ".".join(class_path.split(".")[:-1])
            class_path = class_path.split(".")[-1]

        module = importlib.import_module(module_path)
        class_ = getattr(module, class_path)
        thresholder = class_(**init_args)
        return thresholder
