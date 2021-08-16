"""
Visualizer Callback
"""
from pathlib import Path
from typing import cast
from warnings import warn

from pytorch_lightning import Callback, LightningModule, Trainer
from skimage.segmentation import mark_boundaries
from tqdm import tqdm

from anomalib.datasets.utils import Denormalize
from anomalib.loggers.sigopt import SigoptLogger
from anomalib.models.base import BaseAnomalySegmentationLightning
from anomalib.utils.visualizer import Visualizer


class VisualizerCallback(Callback):
    """
    Callback that visualizes the inference results of a model. The callback generates a figure showing the original
    image, the ground truth segmentation mask, the predicted error heat map, and the predicted segmentation mask.

    To write the images to the Sigopt logger, add the 'sigopt' keyword to the project.log_images_to parameter in the
    config.yaml file. To save the images to the filesystem, add the 'local' keyword.
    """

    def __init__(self):
        super(VisualizerCallback, self).__init__()

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log images at the end of training
        Args:
            trainer (Trainer): Pytorch lightning trainer object
            pl_module (LightningModule): Lightning modules derived from BaseAnomalySegmentationLightning object as
            currently only they support logging images.
        """

        # while this check is in place, this might not work as all modules are subclasses of LightningModule
        # including BaseAnomalySegmentationLightning
        assert isinstance(
            pl_module, BaseAnomalySegmentationLightning
        ), "This callback currently only supports a lightning module of instance BaseAnomalySegmentationLightning"

        module = cast(BaseAnomalySegmentationLightning, pl_module)  # placate mypy

        threshold, _ = module.model.anomaly_map_generator.compute_adaptive_threshold(
            module.true_masks, module.anomaly_maps
        )
        for (filename, image, true_mask, anomaly_map) in tqdm(
            zip(module.filenames, module.images, module.true_masks, module.anomaly_maps),
            desc="Saving Results",
            total=len(module.filenames),
        ):
            image = Denormalize()(image)

            heat_map = module.model.anomaly_map_generator.apply_heatmap_on_image(anomaly_map, image)
            pred_mask = module.model.anomaly_map_generator.compute_mask(anomaly_map, threshold)
            vis_img = mark_boundaries(image, pred_mask, color=(1, 0, 0), mode="thick")

            visualizer = Visualizer(num_rows=1, num_cols=5, figure_size=(12, 3))
            visualizer.add_image(image=image, title="Image")
            visualizer.add_image(image=true_mask, color_map="gray", title="Ground Truth")
            visualizer.add_image(image=heat_map, title="Predicted Heat Map")
            visualizer.add_image(image=pred_mask, color_map="gray", title="Predicted Mask")
            visualizer.add_image(image=vis_img, title="Segmentation Result")

            if "sigopt" in module.hparams.project.log_images_to and isinstance(module.logger, SigoptLogger):
                module.logger.log_image(visualizer.figure, filename.parent.name + "_" + filename.name)
            elif "sigopt" in module.hparams.project.log_images_to and not isinstance(module.logger, SigoptLogger):
                warn(
                    f"Requested SigOpt logging but the logger object is of type: {type(module.logger)}."
                    f"Skipping logging to SigOpt"
                )
            if "local" in module.hparams.project.log_images_to:
                visualizer.save(Path(module.hparams.project.path) / "images" / filename.parent.name / filename.name)
            visualizer.close()
