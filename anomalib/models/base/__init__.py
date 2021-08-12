"""Base classes for all the anomalylib models. All algorithms should inherit the respective base class"""
from anomalib.models.base.lightning_modules import BaseAnomalySegmentationLightning

__all__ = ["BaseAnomalySegmentationLightning"]
