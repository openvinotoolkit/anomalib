"""
Base Anomaly Lightning Models
"""

from pathlib import Path
from typing import List, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor

from anomalib.models.base.torch_modules import BaseAnomalySegmentationModule


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
    Base Anomaly Segmentation Lightning module. All segmentation modules should be derived from this class.
    Any class derived from this module should only wrap the algorithm in the pytorch lightning framework.
    The actual algorithm should be contained within `self.model`.
    """

    def __init__(self, hparams):
        super().__init__(hparams)

        self.filenames: List[Union[str, Path]]
        self.images: List[Union[np.ndarray, Tensor]]

        self.true_masks: List[Union[np.ndarray, Tensor]]
        self.anomaly_maps: List[Union[np.ndarray, Tensor]]

        self.true_labels: List[Union[np.ndarray, Tensor]]
        self.pred_labels: List[Union[np.ndarray, Tensor]]

        self.image_roc_auc: float
        self.pixel_roc_auc: float

        self.image_f1_score: float

        self.model: BaseAnomalySegmentationModule
