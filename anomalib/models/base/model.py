"""
Base Anomaly Models
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from skimage.segmentation import mark_boundaries
from sklearn.metrics import roc_auc_score
from torch import Tensor

from anomalib.datasets.utils import Denormalize
from anomalib.utils.visualizer import Visualizer


class BaseAnomalyLightning(pl.LightningModule):
    """
    BaseAnomalyModel
    """

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.loss: torch.Tensor


class BaseAnomalySegmentationLightning(BaseAnomalyLightning):
    """
    BaseAnomalySegmentationLightning
    """

    def __init__(self, hparams):
        super().__init__(hparams)

        self.filenames: Optional[List[Union[str, Path]]] = None
        self.images: Optional[List[Union[np.ndarray, Tensor]]] = None

        self.true_masks: Optional[List[Union[np.ndarray, Tensor]]] = None
        self.anomaly_maps: Optional[List[Union[np.ndarray, Tensor]]] = None

        self.true_labels: Optional[List[Union[np.ndarray, Tensor]]] = None
        self.pred_labels: Optional[List[Union[np.ndarray, Tensor]]] = None

        self.image_roc_auc: Optional[float] = None
        self.pixel_roc_auc: Optional[float] = None

    def validation_epoch_end(self, outputs):
        """Compute anomaly scores of the validation set, based on the embedding
                        extracted from deep hierarchical CNN features.

        Args:
                outputs: Batch of outputs from the validation step

        Returns:

        """
        self.filenames = [Path(f) for x in outputs for f in x["filenames"]]
        self.images = torch.vstack([x["images"] for x in outputs])

        self.true_masks = np.vstack([x["true_masks"] for x in outputs])
        self.anomaly_maps = np.vstack([x["anomaly_maps"] for x in outputs])

        self.true_labels = np.hstack([x["true_labels"] for x in outputs])
        self.pred_labels = self.anomaly_maps.reshape(self.anomaly_maps.shape[0], -1).max(axis=1)

        self.image_roc_auc = roc_auc_score(self.true_labels, self.pred_labels)
        self.pixel_roc_auc = roc_auc_score(self.true_masks.flatten(), self.anomaly_maps.flatten())

        self.log(name="Image-Level AUC", value=self.image_roc_auc, on_epoch=True, prog_bar=True)
        self.log(name="Pixel-Level AUC", value=self.pixel_roc_auc, on_epoch=True, prog_bar=True)

    def test_epoch_end(self, outputs):
        """
        Compute and save anomaly scores of the test set, based on the embedding
            extracted from deep hierarchical CNN features.

        Args:
            outputs: Batch of outputs from the validation step

        """
        self.validation_epoch_end(outputs)
        threshold = self.anomaly_map_generator.compute_adaptive_threshold(self.true_masks, self.anomaly_maps)

        for (filename, image, true_mask, anomaly_map) in zip(
            self.filenames, self.images, self.true_masks, self.anomaly_maps
        ):
            image = Denormalize()(image.squeeze())

            heat_map = self.anomaly_map_generator.apply_heatmap_on_image(anomaly_map, image)
            pred_mask = self.anomaly_map_generator.compute_mask(anomaly_map=anomaly_map, threshold=threshold)
            vis_img = mark_boundaries(image, pred_mask, color=(1, 0, 0), mode="thick")

            visualizer = Visualizer(num_rows=1, num_cols=5, figure_size=(12, 3))
            visualizer.add_image(image=image, title="Image")
            visualizer.add_image(image=true_mask, color_map="gray", title="Ground Truth")
            visualizer.add_image(image=heat_map, title="Predicted Heat Map")
            visualizer.add_image(image=pred_mask, color_map="gray", title="Predicted Mask")
            visualizer.add_image(image=vis_img, title="Segmentation Result")
            visualizer.save(Path(self.hparams.project.path) / "images" / filename.parent.name / filename.name)
            visualizer.close()
