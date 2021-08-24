"""
Base Anomaly Models
"""
from torch import nn

from anomalib.core.utils.anomaly_map_generator import BaseAnomalyMapGenerator


class BaseAnomalySegmentationModule(nn.Module):
    """Base Anomaly Segmentation module. All segmentation models should be derived from this class"""

    def __init__(self):
        super().__init__()
        self.anomaly_map_generator: BaseAnomalyMapGenerator
