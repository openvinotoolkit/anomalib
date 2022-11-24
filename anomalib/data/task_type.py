"""Task type enum."""

from enum import Enum


class TaskType(str, Enum):
    """Task type used when generating predictions on the dataset."""

    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
