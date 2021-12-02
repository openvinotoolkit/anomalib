"""Visualizer Callback."""
from pathlib import Path
from typing import List
from warnings import warn

from pytorch_lightning import Callback, LightningModule, Trainer
from skimage.segmentation import mark_boundaries
from tqdm import tqdm

from anomalib.core.model import AnomalyModule
from anomalib.core.results import SegmentationResults
from anomalib.data.transforms import Denormalize
from anomalib.loggers import AVAILABLE_LOGGERS
from anomalib.utils.metrics import compute_threshold_and_f1_score
from anomalib.utils.post_process import compute_mask, superimpose_anomaly_map
from anomalib.utils.visualizer import Visualizer


class VisualizerCallback(Callback):
    """Callback that visualizes the inference results of a model.

    The callback generates a figure showing the original image,
    the ground truth segmentation mask, the predicted error heat map,
    and the predicted segmentation mask.

    To save the images to the filesystem, add the 'local' keyword to
    the ``self.loggers`` parameter in the config.yaml file.
    """

    def __init__(self, loggers: List[str]):
        """Visualizer callback."""
        self.loggers = loggers

    def _add_images(
        self,
        visualizer: Visualizer,
        trainer: Trainer,
        module: AnomalyModule,
        filename: Path,
    ):

        # store current logger type as a string
        logger_type = type(module.logger).__name__.lower()

        # save image to respective logger
        for logger in self.loggers:
            if logger in AVAILABLE_LOGGERS:
                # check if logger object is same as the requested object
                if logger in logger_type and module.logger is not None:
                    module.logger.add_image(
                        image=visualizer.figure,
                        name=filename.parent.name + "_" + filename.name,
                        global_step=module.global_step,
                    )
                else:
                    warn(
                        f"Requested {logger} logging but logger object is of type: {type(module.logger)}."
                        f" Skipping logging to {logger}"
                    )

        if "local" in self.loggers:
            if trainer.log_dir:
                visualizer.save(Path(trainer.log_dir) / "images" / filename.parent.name / filename.name)
            else:
                raise ValueError("trainer.log_dir does not exist to save the results.")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log images at the end of training.

        Args:
            trainer (Trainer): Pytorch lightning trainer object (unused)
            pl_module (LightningModule): Lightning modules derived from BaseAnomalyLightning object as
            currently only they support logging images.
        """
        if isinstance(pl_module.results, SegmentationResults):
            results = pl_module.results
        else:
            raise ValueError("Visualizer callback only supported for segmentation tasks.")

        if results.images is None or results.true_masks is None or results.anomaly_maps is None:
            raise ValueError("Result set cannot be empty!")

        threshold, _ = compute_threshold_and_f1_score(results.true_masks, results.anomaly_maps)

        for (filename, image, true_mask, anomaly_map) in tqdm(
            zip(results.filenames, results.images, results.true_masks, results.anomaly_maps),
            desc="Saving Results",
            total=len(results.filenames),
        ):
            image = Denormalize()(image)

            heat_map = superimpose_anomaly_map(anomaly_map, image)
            pred_mask = compute_mask(anomaly_map, threshold)
            vis_img = mark_boundaries(image, pred_mask, color=(1, 0, 0), mode="thick")

            visualizer = Visualizer(num_rows=1, num_cols=5, figure_size=(12, 3))
            visualizer.add_image(image=image, title="Image")
            visualizer.add_image(image=true_mask, color_map="gray", title="Ground Truth")
            visualizer.add_image(image=heat_map, title="Predicted Heat Map")
            visualizer.add_image(image=pred_mask, color_map="gray", title="Predicted Mask")
            visualizer.add_image(image=vis_img, title="Segmentation Result")

            self._add_images(visualizer, trainer, pl_module, filename)

            visualizer.close()
