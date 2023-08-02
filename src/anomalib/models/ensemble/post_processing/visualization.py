"""Class used for visualization of ensemble predictions."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from omegaconf import DictConfig, ListConfig

from anomalib.models.ensemble.post_processing import EnsemblePostProcess
from anomalib.post_processing import Visualizer


class EnsembleVisualization(EnsemblePostProcess):
    """
    Visualize predictions obtained using ensemble.

    Args:
        config: Configurable parameters object, used to set up visualization.
    """

    def __init__(self, config: DictConfig | ListConfig) -> None:
        super().__init__(final_compute=False, name="visualize")
        self.visualizer = Visualizer(mode=config.visualization.mode, task=config.dataset.task)

        self.save = config.visualization.save_images
        self.show = config.visualization.show_images

        image_save_path = config.visualization.image_save_path or config.project.path + "/images"
        self.image_save_path = Path(image_save_path)

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
