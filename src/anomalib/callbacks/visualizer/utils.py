"""Utils to load visualization callbacks."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib import TaskType
from anomalib.utils.visualization import VisualizationMode

from .visualizer_base import BaseVisualizerCallback
from .visualizer_image import ImageVisualizerCallback
from .visualizer_metric import MetricVisualizerCallback


def get_visualization_callbacks(
    task: TaskType,
    image_save_path: str = "",
    mode: VisualizationMode = VisualizationMode.FULL,
    inputs_are_normalized: bool = True,
    show_images: bool = False,
    log_images: bool = True,
    save_images: bool = True,
) -> list[BaseVisualizerCallback]:
    """Get visualization callbacks.

    Args:
        task (TaskType): The task type.
        image_save_path (str): Path to save images.
        mode (VisualizationMode): The visualization mode.
        inputs_are_normalized (bool): Whether the inputs are normalized.
        show_images (bool): Whether to show images.
        log_images (bool): Whether to log images.
        save_images (bool): Whether to save images.

    Returns:
        List of visualization callbacks.
    """
    return [
        callback(
            task=task,
            mode=mode,
            image_save_path=image_save_path,
            inputs_are_normalized=inputs_are_normalized,
            show_images=show_images,
            log_images=log_images,
            save_images=save_images,
        )
        for callback in (ImageVisualizerCallback, MetricVisualizerCallback)
    ]
