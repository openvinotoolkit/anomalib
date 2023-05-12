"""Manages visualization."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import Any, cast

import numpy as np
from matplotlib import pyplot as plt
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

from anomalib import trainer
from anomalib.data import TaskType
from anomalib.models.components import AnomalyModule
from anomalib.post_processing import VisualizationMode
from anomalib.post_processing import Visualizer as _Visualizer
from anomalib.utils.loggers.base import ImageLoggerBase
from anomalib.utils.metrics.collection import AnomalibMetricCollection


class VisualizationStage(str, Enum):
    """Visualization stage."""

    VAL = "val"
    TEST = "test"
    PREDICT = "predict"


class Visualizer:
    """Manages visualization.

    Args:
        trainer (core.AnomalibTrainer): Anomaly trainer.
        mode (VisualizationMode): The mode of visualization. Can be one of ['full', 'simple'].
        show_images (bool, optional): Whether to show images. Defaults to False.
        log_images (bool, optional): Whether to log images to available loggers. Defaults to False.
        stage (VisualizationStage, optional): The stage at which to write images to the logger(s).
            Defaults to VisualizationStage.TEST.
    """

    def __init__(
        self,
        trainer: trainer.AnomalibTrainer,
        mode: VisualizationMode,
        show_images: bool = False,
        log_images: bool = False,
        stage: VisualizationStage = VisualizationStage.TEST,
    ) -> None:
        if mode not in set(VisualizationMode):
            raise ValueError(f"Unknown visualization mode: {mode}. Please choose one of {set(VisualizationMode)}")
        self.mode = mode
        if trainer.task_type not in (TaskType.CLASSIFICATION, TaskType.DETECTION, TaskType.SEGMENTATION):
            raise ValueError(
                f"Unknown task type: {mode}. Please choose one of ['classification', 'detection', 'segmentation']"
            )
        self.trainer = trainer
        self.show_images = show_images
        self.log_images = log_images
        self.stage = stage
        self.visualizer = _Visualizer(mode, trainer.task_type)

    def visualize_images(
        self, outputs: EPOCH_OUTPUT | list[EPOCH_OUTPUT] | STEP_OUTPUT, stage: VisualizationStage
    ) -> None:
        """Visualize or show the outputs.

        Args:
            outputs (EPOCH_OUTPUT, List[EPOCH_OUTPUT]): The outputs to visualize.
            stage (str): The stage at which to visualize.
        """
        if stage == self.stage and (self.show_images or self.log_images):
            if isinstance(outputs, list):
                for output in outputs:
                    self.visualize_images(output, stage)
            else:
                for i, image in enumerate(self.visualizer.visualize_batch(outputs)):
                    filename = self._get_filename(outputs, i)
                    if self.show_images:
                        self.visualizer.show(str(filename), image)
                    if self.log_images:
                        self._add_to_loggers(image, filename=filename)

    def visualize_metrics(self, stage: VisualizationStage, metrics_list: list[AnomalibMetricCollection]) -> None:
        """Visualize metrics.

        Note:
            This should only be called after the metrics have been computed. Otherwise, it will log incorrect metrics.

        Args:
            stage (VisualizationStage): The stage at which to visualize metrics.
        """
        if stage == self.stage and (self.show_images or self.log_images):
            for metrics in metrics_list:
                for metric in metrics.values():
                    # `generate_figure` needs to be defined for every metric that should be plotted automatically
                    if hasattr(metric, "generate_figure"):
                        fig, log_name = metric.generate_figure()
                        file_name = f"{metrics.prefix}{log_name}"
                        if self.log_images:
                            self._add_to_loggers(fig, filename=file_name)
                        if self.show_images:
                            # TODO: test this
                            self.visualizer.show(file_name, fig)
                        plt.close(fig)

    def _get_filename(self, outputs: Any, index: int) -> Path:
        """Gets file name from the outputs corresponding to the index.

        Args:
            outputs (Any): Outputs.
            index (int): Index of the image.

        Raises:
            KeyError: If neither ``image_path`` nor ``video_path`` is present in the outputs.
        """
        if "image_path" in outputs.keys():
            filename = Path(outputs["image_path"][index])
        elif "video_path" in outputs.keys():
            zero_fill = int(math.log10(outputs["last_frame"][index])) + 1
            suffix = f"{str(outputs['frames'][index].int().item()).zfill(zero_fill)}.png"
            filename = Path(outputs["video_path"][index]) / suffix
        else:
            raise KeyError("Batch must have either 'image_path' or 'video_path' defined.")
        return filename

    @property
    def anomaly_module(self) -> AnomalyModule:
        """Returns anomaly module.

        We can't directly access the anomaly module in ``__init__`` because it is not available till it is passed to the
        trainer.
        """
        return self.trainer.lightning_module

    def _add_to_loggers(
        self,
        image: np.ndarray,
        filename: str | Path,
    ) -> None:
        """Log image from a visualizer to each of the available loggers in the project.

        Args:
            image (np.ndarray): Image that should be added to the loggers.
            filename (Path): Path of the input image. This name is used as name for the generated image.
        """
        # Store names of logger and the logger in a dict
        available_loggers = {
            type(logger).__name__.lower().rstrip("logger").lstrip("anomalib"): logger for logger in self.trainer.loggers
        }
        # save image to respective logger
        if self.log_images:
            for available_logger in available_loggers.values():
                # check if logger object is same as the requested object
                if isinstance(available_logger, ImageLoggerBase):
                    logger: ImageLoggerBase = cast(ImageLoggerBase, available_logger)  # placate mypy
                    if isinstance(filename, Path):
                        _name = filename.parent.name + "_" + filename.name
                    elif isinstance(filename, str):
                        _name = filename
                    logger.add_image(
                        image=image,
                        name=_name,
                        global_step=self.anomaly_module.global_step,
                    )
