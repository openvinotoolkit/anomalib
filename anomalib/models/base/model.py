"""
Base Anomaly Models
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor


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

        self.image_f1_score: Optional[float] = None
