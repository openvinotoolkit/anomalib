"""
Base Anomaly Models
"""
from torch import nn

from anomalib.core.utils.anomaly_map_generator import BaseAnomalyMapGenerator


class BaseAnomalyModule(nn.Module):
    """Base Anomaly module. All image anomaly models should be derived from this class"""

    def __init__(self):
        super().__init__()
        self.anomaly_map_generator: BaseAnomalyMapGenerator

    def forward(self, _):
        """
        To be implemented in the subclass
        """
        raise NotImplementedError
