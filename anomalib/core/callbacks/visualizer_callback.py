"""Visualizer Callback."""
from pathlib import Path
from typing import Any, Optional
from warnings import warn

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from skimage.segmentation import mark_boundaries

from anomalib import loggers
from anomalib.core.model import AnomalyModule
from anomalib.data.transforms import Denormalize
from anomalib.loggers.wandb import AnomalibWandbLogger
from anomalib.utils.post_process import compute_mask, superimpose_anomaly_map
from anomalib.utils.visualizer import Visualizer


class VisualizerCallback(Callback):
    """Callback that visualizes the inference results of a model.

    The callback generates a figure showing the original image, the ground truth segmentation mask,
    the predicted error heat map, and the predicted segmentation mask.

    To save the images to the filesystem, add the 'local' keyword to the `project.log_images_to` parameter in the
    config.yaml file.
    """

    def __init__(self, inputs_are_normalized: bool = True):
        """Visualizer callback."""
        self.inputs_are_normalized = inputs_are_normalized

    def _add_images(
        self,
        visualizer: Visualizer,
        module: AnomalyModule,
        filename: Path,
    ):
        """Save image to logger/local storage.

        Saves the image in `visualizer.figure` to the respective loggers and local storage if specified in
        `log_images_to` in `config.yaml` of the models.

        Args:
            visualizer (Visualizer): Visualizer object from which the `figure` is saved/logged.
            module (AnomalyModule): Anomaly module which holds reference to `hparams` and `logger`.
            filename (Path): Path of the input image. This name is used as name for the generated image.
        """

        # store current logger type as a string
        logger_type = type(module.logger).__name__.lower()

        # save image to respective logger
        for log_to in module.hparams.project.log_images_to:
            if log_to in loggers.AVAILABLE_LOGGERS:
                # check if logger object is same as the requested object
                if log_to in logger_type and module.logger is not None:
                    module.logger.add_image(
                        image=visualizer.figure,
                        name=filename.parent.name + "_" + filename.name,
                        global_step=module.global_step,
                    )
                else:
                    warn(
                        f"Requested {log_to} logging but logger object is of type: {type(module.logger)}."
                        f" Skipping logging to {log_to}"
                    )

        if "local" in module.hparams.project.log_images_to:
            visualizer.save(Path(module.hparams.project.path) / "images" / filename.parent.name / filename.name)

    def on_test_batch_end(
        self,
        _trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Log images at the end of every batch.

        Args:
            _trainer (Trainer): Pytorch lightning trainer object (unused).
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
        threshold = pl_module.pixel_metrics.F1.threshold

        for (filename, image, true_mask, anomaly_map) in zip(
            outputs["image_path"], outputs["image"], outputs["mask"], outputs["anomaly_maps"]
        ):
            image = Denormalize()(image.cpu())
            true_mask = true_mask.cpu().numpy()
            anomaly_map = anomaly_map.cpu().numpy()

            heat_map = superimpose_anomaly_map(anomaly_map, image, normalize=normalize)
            pred_mask = compute_mask(anomaly_map, threshold)
            vis_img = mark_boundaries(image, pred_mask, color=(1, 0, 0), mode="thick")

            visualizer = Visualizer(num_rows=1, num_cols=5, figure_size=(12, 3))
            visualizer.add_image(image=image, title="Image")
            visualizer.add_image(image=true_mask, color_map="gray", title="Ground Truth")
            visualizer.add_image(image=heat_map, title="Predicted Heat Map")
            visualizer.add_image(image=pred_mask, color_map="gray", title="Predicted Mask")
            visualizer.add_image(image=vis_img, title="Segmentation Result")
            self._add_images(visualizer, pl_module, Path(filename))
            visualizer.close()

    def on_test_end(self, _trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Sync logs.

        Currently only ``AnomalibWandbLogger`` is called from this method. This is because logging as a single batch
        ensures that all images appear as part of the same step.

        Args:
            _trainer (pl.Trainer): Pytorch Lightning trainer (unused)
            pl_module (pl.LightningModule): Anomaly module
        """
        if pl_module.logger is not None and isinstance(pl_module.logger, AnomalibWandbLogger):
            pl_module.logger.save()
