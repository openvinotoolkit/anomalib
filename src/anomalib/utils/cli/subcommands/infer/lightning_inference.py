"""Provides Lightning inference."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from jsonargparse import ArgumentParser
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.post_processing.visualizer import VisualizationMode
from anomalib.trainer import AnomalibTrainer
from anomalib.utils.callbacks import get_callbacks
from anomalib.utils.loggers import get_experiment_logger


def get_lightning_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to a config file")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--input", type=Path, required=True, help="Path to image(s) to infer.")
    parser.add_argument("--output", type=str, required=False, help="Path to save the output image(s).")
    parser.add_argument(
        "--visualization_mode",
        type=VisualizationMode,
        required=False,
        default=VisualizationMode.SIMPLE,
        help="Visualization mode.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        required=False,
        help="Show the visualized predictions on the screen.",
    )

    return parser


class LightningInference:
    def __init__(
        self, config: Path, weights: Path, input: Path, output: str, visualization_mode: VisualizationMode, show: bool
    ):
        self.weights = weights
        self.input = input
        self.output = output
        self.visualization_mode = visualization_mode
        self.show = show
        self.config = get_configurable_parameters(config_path=config)
        self._update_config()

        # create model and trainer
        self.model = get_model(self.config)
        callbacks = get_callbacks(self.config)
        loggers = get_experiment_logger(self.config)

        self.trainer = AnomalibTrainer(
            **self.config.trainer,
            **self.config.post_processing,
            **self.config.visualization,
            logger=loggers,
            callbacks=callbacks,
            image_metrics=self.config.metrics.get("image", None),
            pixel_metrics=self.config.metrics.get("pixel", None)
        )

        self.dataloader = self._get_dataloader()

    def _get_dataloader(self) -> DataLoader:
        # get the transforms
        transform_config = (
            self.config.dataset.transform_config.eval if "transform_config" in self.config.dataset.keys() else None
        )
        image_size = (self.config.dataset.image_size[0], self.config.dataset.image_size[1])
        center_crop = self.config.dataset.get("center_crop")
        if center_crop is not None:
            center_crop = tuple(center_crop)
        normalization = InputNormalizationMethod(self.config.dataset.normalization)
        transform = get_transforms(
            config=transform_config, image_size=image_size, center_crop=center_crop, normalization=normalization
        )

        # create the dataset
        dataset = InferenceDataset(
            self.input, image_size=tuple(self.config.dataset.image_size), transform=transform  # type: ignore
        )
        return DataLoader(dataset)

    def _update_config(self):
        self.config.visualization.show_images = self.show
        self.config.visualization.visualization_mode = self.visualization_mode
        if self.output:  # set FileSystemLogger and it's path
            self.config.logging.loggers = {
                "class_path": "anomalib.utils.loggers.file_system.FileSystemLogger",
                "init_args": {"save_dir": self.output},
            }
            self.config.visualization.log_images = True
            self.config.visualization.visualization_stage = "predict"
        else:
            # empty loggers so that images are not saved
            self.config.logging.loggers = None
            self.config.visualization.log_images = False

    def run(self):
        self.trainer.predict(model=self.model, dataloaders=[self.dataloader], ckpt_path=str(self.weights))
