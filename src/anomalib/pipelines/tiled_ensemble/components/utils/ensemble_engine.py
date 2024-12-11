"""Implements custom Anomalib engine for tiled ensemble training."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

from lightning.pytorch.callbacks import Callback, RichModelSummary

from anomalib.callbacks import ModelCheckpoint, TimerCallback
from anomalib.callbacks.metrics import _MetricsCallback
from anomalib.callbacks.normalization import get_normalization_callback
from anomalib.callbacks.post_processor import _PostProcessorCallback
from anomalib.callbacks.thresholding import _ThresholdCallback
from anomalib.engine import Engine
from anomalib.models import AnomalyModule
from anomalib.utils.path import create_versioned_dir

logger = logging.getLogger(__name__)


class TiledEnsembleEngine(Engine):
    """Engine used for training and evaluating tiled ensemble.

    Most of the logic stays the same, but workspace creation and callbacks are adjusted for ensemble.

    Args:
        tile_index (tuple[int, int]): index of tile that this engine instance processes.
        **kwargs: Engine arguments.
    """

    def __init__(self, tile_index: tuple[int, int], **kwargs) -> None:
        self.tile_index = tile_index
        super().__init__(**kwargs)

    def _setup_workspace(self, *args, **kwargs) -> None:
        """Skip since in case of tiled ensemble, workspace is only setup once at the beginning of training."""

    @staticmethod
    def setup_ensemble_workspace(args: dict, versioned_dir: bool = True) -> Path:
        """Set up the workspace at the beginning of tiled ensemble training.

        Args:
            args (dict): Tiled ensemble config dict.
            versioned_dir (bool, optional): Whether to create a versioned directory.
                Defaults to ``True``.

        Returns:
            Path: path to new workspace root dir
        """
        model_name = args["TrainModels"]["model"]["class_path"].split(".")[-1]
        dataset_name = args["data"]["class_path"].split(".")[-1]
        category = args["data"]["init_args"]["category"]
        root_dir = Path(args["default_root_dir"]) / model_name / dataset_name / category
        return create_versioned_dir(root_dir) if versioned_dir else root_dir / "latest"

    def _setup_anomalib_callbacks(self, model: AnomalyModule) -> None:
        """Modified method to enable individual model training. It's called when Trainer is being set up."""
        del model  # not used here

        _callbacks: list[Callback] = [RichModelSummary()]

        # Add ModelCheckpoint if it is not in the callbacks list.
        has_checkpoint_callback = any(isinstance(c, ModelCheckpoint) for c in self._cache.args["callbacks"])
        if not has_checkpoint_callback:
            tile_i, tile_j = self.tile_index
            _callbacks.append(
                ModelCheckpoint(
                    dirpath=self._cache.args["default_root_dir"] / "weights" / "lightning",
                    filename=f"model{tile_i}_{tile_j}",
                    auto_insert_metric_name=False,
                ),
            )

        # Add the post-processor callbacks. Used for thresholding and label calculation.
        _callbacks.append(_PostProcessorCallback())

        # Add the  normalization callback if tile level normalization was specified (is not none).
        normalization_callback = get_normalization_callback(self.normalization)
        if normalization_callback is not None:
            _callbacks.append(normalization_callback)

        # Add the thresholding and metrics callbacks in all cases,
        # because individual model might still need this for early stop.
        _callbacks.append(_ThresholdCallback(self.threshold))
        _callbacks.append(_MetricsCallback(self.task, self.image_metric_names, self.pixel_metric_names))

        _callbacks.append(TimerCallback())

        # Combine the callbacks, and update the trainer callbacks.
        self._cache.args["callbacks"] = _callbacks + self._cache.args["callbacks"]
