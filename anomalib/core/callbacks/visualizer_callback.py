"""Visualizer Callback."""
from pathlib import Path
from warnings import warn

from pytorch_lightning import Callback, LightningModule, Trainer
from skimage.segmentation import mark_boundaries
from tqdm import tqdm

from anomalib import loggers
from anomalib.core.model import AnomalyModule
from anomalib.core.results import SegmentationResults
from anomalib.datasets.utils import Denormalize
from anomalib.utils.metrics import compute_threshold_and_f1_score
from anomalib.utils.post_process import compute_mask, superimpose_anomaly_map
from anomalib.utils.visualizer import Visualizer


class VisualizerCallback(Callback):
    """Callback that visualizes the inference results of a model.

    The callback generates a figure showing the original image, the ground truth segmentation mask,
    the predicted error heat map, and the predicted segmentation mask.

    To save the images to the filesystem, add the 'local' keyword to the `project.log_images_to` parameter in the
    config.yaml file.
    """

    def __init__(self):
        """Visualizer callback."""

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

    def on_test_epoch_end(self, _trainer: Trainer, pl_module: LightningModule) -> None:
        """Log images at the end of training.

        Args:
            _trainer (Trainer): Pytorch lightning trainer object (unused)
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
            self._add_images(visualizer, pl_module, filename)
            visualizer.close()
