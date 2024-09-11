"""Thresholding callback."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import Any

import torch
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig, ListConfig

from anomalib.metrics.threshold import Threshold
from anomalib.models import AnomalyModule
from anomalib.utils.types import THRESHOLD


class _ThresholdCallback(Callback):
    """Setup/apply thresholding.

    Note: This callback is set within the Engine.
    """

    def __init__(
        self,
        threshold: THRESHOLD = "F1AdaptiveThreshold",
    ) -> None:
        super().__init__()
        self._initialize_thresholds(threshold)
        self.image_threshold: Threshold
        self.pixel_threshold: Threshold

    def setup(self, trainer: Trainer, pl_module: AnomalyModule, stage: str) -> None:
        del trainer, stage  # Unused arguments.
        if not hasattr(pl_module, "image_threshold"):
            pl_module.image_threshold = self.image_threshold
        if not hasattr(pl_module, "pixel_threshold"):
            pl_module.pixel_threshold = self.pixel_threshold

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: AnomalyModule) -> None:
        del trainer  # Unused argument.
        self._reset(pl_module)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, batch, batch_idx, dataloader_idx  # Unused arguments.
        if outputs is not None:
            self._outputs_to_cpu(outputs)
            self._update(pl_module, outputs)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: AnomalyModule) -> None:
        del trainer  # Unused argument.
        self._compute(pl_module)

    def _initialize_thresholds(
        self,
        threshold: THRESHOLD,
    ) -> None:
        """Initialize ``self.image_threshold`` and ``self.pixel_threshold``.

        Args:
            threshold (THRESHOLD):
                Threshold configuration

        Example:
            >>> _initialize_thresholds(F1AdaptiveThreshold())
            or
            >>> _initialize_thresholds((ManualThreshold(0.5), ManualThreshold(0.5)))
            or configuration

        For more details on configuration see :fun:`_load_from_config`

        Raises:
            ValueError: Unknown threshold class or incorrect configuration
        """
        # TODO(djdameln): Add tests for each case
        # CVS-122661
        # When only a single threshold class is passed.
        # This initializes image and pixel thresholds with the same class
        # >>> _initialize_thresholds(F1AdaptiveThreshold())
        if isinstance(threshold, Threshold):
            self.image_threshold = threshold
            self.pixel_threshold = threshold.clone()

        # When a tuple of threshold classes are passed
        # >>> _initialize_thresholds((ManualThreshold(0.5), ManualThreshold(0.5)))
        elif isinstance(threshold, tuple) and isinstance(threshold[0], Threshold):
            self.image_threshold = threshold[0]
            self.pixel_threshold = threshold[1]
        # When the passed threshold is not an instance of a Threshold class.
        elif isinstance(threshold, str | DictConfig | ListConfig | list):
            self._load_from_config(threshold)
        else:
            msg = f"Invalid threshold type {type(threshold)}"
            raise TypeError(msg)

    def _load_from_config(self, threshold: DictConfig | str | ListConfig | list[dict[str, str | float]]) -> None:
        """Load the thresholding class based on the config.

        Example:
            threshold: F1AdaptiveThreshold
            or
            threshold:
                class_path: F1AdaptiveThreshold
                init_args:
                    -
            or
            threshold:
                - F1AdaptiveThreshold
                - F1AdaptiveThreshold
            or
            threshold:
                - class_path: F1AdaptiveThreshold
                    init_args:
                        -
                - class_path: F1AdaptiveThreshold
        """
        if isinstance(threshold, str | DictConfig):
            self.image_threshold = self._get_threshold_from_config(threshold)
            self.pixel_threshold = self.image_threshold.clone()
        elif isinstance(threshold, ListConfig | list):
            self.image_threshold = self._get_threshold_from_config(threshold[0])
            self.pixel_threshold = self._get_threshold_from_config(threshold[1])
        else:
            msg = f"Invalid threshold config {threshold}"
            raise TypeError(msg)

    @staticmethod
    def _get_threshold_from_config(threshold: DictConfig | str | dict[str, str | float]) -> Threshold:
        """Return the instantiated threshold object.

        Example:
            >>> _get_threshold_from_config(F1AdaptiveThreshold)
            or
            >>> config = DictConfig({
            ...    "class_path": "ManualThreshold",
            ...    "init_args": {"default_value": 0.7}
            ... })
            >>> __get_threshold_from_config(config)
            or
            >>> config = DictConfig({
            ...    "class_path": "anomalib.metrics.threshold.F1AdaptiveThreshold"
            ... })
            >>> __get_threshold_from_config(config)

        Returns:
            (Threshold): Instance of threshold object.
        """
        if isinstance(threshold, str):
            threshold = DictConfig({"class_path": threshold})

        class_path = threshold["class_path"]
        init_args = threshold.get("init_args", {})

        if len(class_path.split(".")) == 1:
            module_path = "anomalib.metrics.threshold"

        else:
            module_path = ".".join(class_path.split(".")[:-1])
            class_path = class_path.split(".")[-1]

        module = importlib.import_module(module_path)
        class_ = getattr(module, class_path)
        return class_(**init_args)

    @staticmethod
    def _reset(pl_module: AnomalyModule) -> None:
        pl_module.image_threshold.reset()
        pl_module.pixel_threshold.reset()

    def _outputs_to_cpu(self, output: STEP_OUTPUT) -> STEP_OUTPUT | dict[str, Any]:
        if isinstance(output, dict):
            for key, value in output.items():
                output[key] = self._outputs_to_cpu(value)
        elif isinstance(output, torch.Tensor):
            output = output.cpu()
        return output

    @staticmethod
    def _update(pl_module: AnomalyModule, outputs: STEP_OUTPUT) -> None:
        pl_module.image_threshold.cpu()
        pl_module.image_threshold.update(outputs["pred_scores"], outputs["label"].int())
        if "mask" in outputs and "anomaly_maps" in outputs:
            pl_module.pixel_threshold.cpu()
            pl_module.pixel_threshold.update(outputs["anomaly_maps"], outputs["mask"].int())

    @staticmethod
    def _compute(pl_module: AnomalyModule) -> None:
        pl_module.image_threshold.compute()
        if pl_module.pixel_threshold._update_called:  # noqa: SLF001
            pl_module.pixel_threshold.compute()
        else:
            pl_module.pixel_threshold.value = pl_module.image_threshold.value
