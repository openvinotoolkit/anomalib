import lightning as L
import torch

from anomalib.data import TaskType
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.post_processing.visualizer import VisualizationMode
from anomalib.trainer.connectors import ThresholdingConnector
from anomalib.utils.metrics import AnomalyScoreThreshold


class OneClass:
    def __init__(
        self,
        task_type: TaskType = TaskType.SEGMENTATION,
        threshold_method: ThresholdMethod = ThresholdMethod.ADAPTIVE,
        normalization_method: NormalizationMethod = NormalizationMethod.MIN_MAX,
        visualization_mode: VisualizationMode = VisualizationMode.FULL,
    ):
        self.image_threshold = AnomalyScoreThreshold().cpu()
        self.pixel_threshold = AnomalyScoreThreshold().cpu()

        self.thresholding_connector = ThresholdingConnector(
            pixel_threshold=self.pixel_threshold,
            image_threshold=self.image_threshold,
            threshold_method=threshold_method,
            task_type=task_type,
        )

    def fit(
        self,
        model: L.LightningModule,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ):
        model.fit(train_loader, val_loader)

        self.thresholding_connector.compute()
        self.thresholding_connector.update(None)

        return model
