"""Class used for visualization of ensemble predictions."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from tools.tiled_ensemble.post_processing.postprocess import EnsemblePostProcess

from anomalib.data import TaskType
from anomalib.post_processing import VisualizationMode, Visualizer


class EnsembleVisualization(EnsemblePostProcess):
    """
    Visualize predictions obtained using ensemble.

    Args:
        mode (VisualizationMode): Mode of visualisation (simple, full).
        task (TaskType): Task of current run.
        save_images (bool): Flag to indicating if images need to be saved.
        show_images (bool): Flag indicating if images will be shown.
        save_path (str): Path to where images will be saved.
    """

    def __init__(
        self, mode: VisualizationMode, task: TaskType, save_images: bool, show_images: bool, save_path: str
    ) -> None:
        super().__init__(final_compute=False, name="visualize")
        self.visualizer = Visualizer(mode=mode, task=task)

        self.save = save_images
        self.show = show_images

        self.image_save_path = Path(save_path)

    def process(self, data: dict) -> dict:
        """
        Visualize joined predictions using Visualizer class.

        Args:
            data: Batch of predictions.

        Returns:
            Unchanged input data.
        """
        for i, image in enumerate(self.visualizer.visualize_batch(data)):
            filename = Path(data["image_path"][i])
            if self.save:
                file_path = self.image_save_path / filename.parent.name / filename.name
                self.visualizer.save(file_path, image)
            if self.show:
                self.visualizer.show(str(filename), image)

        return data
