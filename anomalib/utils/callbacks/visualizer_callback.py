"""Visualizer Callback."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from pathlib import Path
from typing import Any, List, Optional, cast
from warnings import warn

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from skimage.segmentation import mark_boundaries

from anomalib.models.components import AnomalyModule
from anomalib.post_processing import (
    Visualizer,
    add_anomalous_label,
    add_normal_label,
    compute_mask,
    superimpose_anomaly_map,
)
from anomalib.pre_processing.transforms import Denormalize
from anomalib.utils import loggers
from anomalib.utils.loggers import AnomalibWandbLogger
from anomalib.utils.loggers.base import ImageLoggerBase


class VisualizerCallback(Callback):
    """Callback that visualizes the inference results of a model.

    The callback generates a figure showing the original image, the ground truth segmentation mask,
    the predicted error heat map, and the predicted segmentation mask.

    To save the images to the filesystem, add the 'local' keyword to the `project.log_images_to` parameter in the
    config.yaml file.
    """

    def __init__(self, task: str, log_images_to: Optional[List[str]] = None, inputs_are_normalized: bool = True):
        """Visualizer callback."""
        self.task = task
        self.log_images_to = [] if log_images_to is None else log_images_to
        self.inputs_are_normalized = inputs_are_normalized

    def _add_images(
        self,
        visualizer: Visualizer,
        module: AnomalyModule,
        trainer: pl.Trainer,
        filename: Path,
    ):
        """Save image to logger/local storage.

        Saves the image in `visualizer.figure` to the respective loggers and local storage if specified in
        `log_images_to` in `config.yaml` of the models.

        Args:
            visualizer (Visualizer): Visualizer object from which the `figure` is saved/logged.
            module (AnomalyModule): Anomaly module.
            trainer (Trainer): Pytorch Lightning trainer which holds reference to `logger`
            filename (Path): Path of the input image. This name is used as name for the generated image.
        """
        # Store names of logger and the logger in a dict
        available_loggers = {
            type(logger).__name__.lower().rstrip("logger").lstrip("anomalib"): logger for logger in trainer.loggers
        }
        # save image to respective logger
        for log_to in self.log_images_to:
            if log_to in loggers.AVAILABLE_LOGGERS:
                # check if logger object is same as the requested object
                if log_to in available_loggers and isinstance(available_loggers[log_to], ImageLoggerBase):
                    logger: ImageLoggerBase = cast(ImageLoggerBase, available_loggers[log_to])  # placate mypy
                    logger.add_image(
                        image=visualizer.figure,
                        name=filename.parent.name + "_" + filename.name,
                        global_step=module.global_step,
                    )
                else:
                    warn(
                        f"Requested {log_to} logging but logger object is of type: {type(module.logger)}."
                        f" Skipping logging to {log_to}"
                    )
            else:
                warn(f"{log_to} not in the list of supported image loggers.")

        if "local" in self.log_images_to:
            visualizer.save(Path(trainer.default_root_dir) / "images" / filename.parent.name / filename.name)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: Optional[STEP_OUTPUT],
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Log images at the end of every batch.

        Args:
            trainer (Trainer): Pytorch lightning trainer object (unused).
            pl_module (LightningModule): Lightning modules derived from BaseAnomalyLightning object as
            currently only they support logging images.
            outputs (Dict[str, Any]): Outputs of the current test step.
            _batch (Any): Input batch of the current test step (unused).
            _batch_idx (int): Index of the current test batch (unused).
            _dataloader_idx (int): Index of the dataloader that yielded the current batch (unused).
        """
        assert outputs is not None

        if self.inputs_are_normalized:
            normalize = False  # anomaly maps are already normalized
        else:
            normalize = True  # raw anomaly maps. Still need to normalize

        threshold = pl_module.pixel_metrics.threshold
        for i, (filename, image, anomaly_map, pred_score, gt_label) in enumerate(
            zip(
                outputs["image_path"],
                outputs["image"],
                outputs["anomaly_maps"],
                outputs["pred_scores"],
                outputs["label"],
            )
        ):
            image = Denormalize()(image.cpu())
            anomaly_map = anomaly_map.cpu().numpy()
            heat_map = superimpose_anomaly_map(anomaly_map, image, normalize=normalize)
            pred_mask = compute_mask(anomaly_map, threshold)
            vis_img = mark_boundaries(image, pred_mask, color=(1, 0, 0), mode="thick")

            visualizer = Visualizer()

            if self.task == "segmentation":
                visualizer.add_image(image=image, title="Image")
                if "mask" in outputs:
                    true_mask = outputs["mask"][i].cpu().numpy() * 255
                    visualizer.add_image(image=true_mask, color_map="gray", title="Ground Truth")
                visualizer.add_image(image=heat_map, title="Predicted Heat Map")
                visualizer.add_image(image=pred_mask, color_map="gray", title="Predicted Mask")
                visualizer.add_image(image=vis_img, title="Segmentation Result")
            elif self.task == "classification":
                gt_im = add_anomalous_label(image) if gt_label else add_normal_label(image)
                visualizer.add_image(gt_im, title="Image/True label")
                if pred_score >= threshold:
                    image_classified = add_anomalous_label(heat_map, pred_score)
                else:
                    image_classified = add_normal_label(heat_map, 1 - pred_score)
                visualizer.add_image(image=image_classified, title="Prediction")

            visualizer.generate()
            self._add_images(visualizer, pl_module, trainer, Path(filename))
            visualizer.close()

    def on_test_end(self, _trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Sync logs.

        Currently only ``AnomalibWandbLogger`` is called from this method. This is because logging as a single batch
        ensures that all images appear as part of the same step.

        Args:
            _trainer (pl.Trainer): Pytorch Lightning trainer (unused)
            pl_module (AnomalyModule): Anomaly module
        """
        if pl_module.logger is not None and isinstance(pl_module.logger, AnomalibWandbLogger):
            pl_module.logger.save()
