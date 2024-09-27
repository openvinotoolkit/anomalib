"""Base classes for metrics in Anomalib."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from torchmetrics import Metric, MetricCollection

from anomalib.data import Batch
from typing import Callable


class AnomalibMetric:
    """Base class for metrics in Anomalib.

    This class is designed to make any torchmetrics metric compatible with the
    Anomalib framework. An Anomalib version of any torchmetrics metric can be created 
    by inheriting from this class and the desired torchmetrics metric. For example, to 
    create an Anomalib version of the BinaryF1Score metric, the user can create a new
    class that inherits from AnomalibMetric and BinaryF1Score.

    The AnomalibMetric class adds the ability to update the metric with a Batch
    object instead of individual prediction and target tensors. To use this feature, 
    the user must provide a list of fields as constructor arguments when instantiating 
    the metric. When the metric is updated with a Batch object, it will extract the 
    values of these fields from the Batch object and pass them to the `update` method 
    of the metric.

    Args:
        fields (Sequence[str]): List of field names to extract from the Batch object.
        prefix (str): Prefix to add to the metric name. Defaults to an empty string.
        **kwargs: Variable keyword arguments that can be passed to the parent class.

    Examples:
        >>> from torchmetrics.classification import BinaryF1Score
        >>> from anomalib.metrics import AnomalibMetric
        >>> from anomalib.data import ImageBatch
        >>> import torch
        >>> 
        >>> class F1Score(AnomalibMetric, BinaryF1Score):
        ...     pass
        ...
        >>> f1_score = F1Score(fields=["pred_label", "gt_label"])
        >>>
        >>> batch = ImageBatch(
        ...     image=torch.rand(4, 3, 256, 256),
        ...     pred_label=torch.tensor([0, 0, 0, 1]),
        ...     gt_label=torch.tensor([0, 0, 1, 1])),
        ... )
        >>> 
        >>> # The AnomalibMetric class allows us to update the metric by passing a Batch
        >>> # object directly.
        >>> f1_score.update(batch)
        >>> f1_score.compute()
        tensor(0.6667)
        >>> 
        >>> # specifying the field names allows us to distinguish between image and 
        >>> # pixel metrics.
        >>> image_f1_score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
        >>> pixel_f1_score = F1Score(fields=[pred_mask", "gt_mask"], prefix="pixel_")
    """

    def __init__(self, fields: Sequence[str], prefix: str = "", **kwargs) -> None:
        self.fields = fields
        self.name = prefix + self.__class__.__name__
        super().__init__(**kwargs)

    def __init_subclass__(cls, **kwargs) -> None:
        """Check that the subclass implements the torchmetrics.Metric interface."""
        del kwargs
        assert issubclass(
            cls, (Metric, MetricCollection)
        ), "AnomalibMetric must be a subclass of torchmetrics.Metric or torchmetrics.MetricCollection"

    def update(self, batch: Batch, *args, **kwargs):
        """Update the metric with the specified fields from the Batch object."""
        values = [getattr(batch, key) for key in self.fields]
        super().update(*values, *args, **kwargs)


def create_anomalib_metric(metric_cls: Callable):
    """Create an Anomalib version of a torchmetrics metric.
    
    This function creates an Anomalib version of a torchmetrics metric by inheriting
    from the AnomalibMetric class and the specified torchmetrics metric class. The
    resulting class will have the same name as the input metric class and will inherit
    from both AnomalibMetric and the input metric class.

    Args:
        metric_cls (Callable): The torchmetrics metric class to wrap.
    
    Returns:
        AnomalibMetric: An Anomalib version of the input metric class.

    Examples:
        >>> from torchmetrics.classification import BinaryF1Score
        >>> from anomalib.metrics import create_anomalib_metric
        >>> 
        >>> F1Score = create_anomalib_metric(BinaryF1Score)
        >>> # This is equivalent to the following class definition:
        >>> # class F1Score(AnomalibMetric, BinaryF1Score): ...
        >>> 
        >>> f1_score = F1Score(fields=["pred_label", "gt_label"])
        >>> 
        >>> # The AnomalibMetric class allows us to update the metric by passing a Batch
        >>> # object directly.
        >>> f1_score.update(batch)
        >>> f1_score.compute()
        tensor(0.6667)
    """
    assert issubclass(metric_cls, Metric), "The wrapped metric must be a subclass of torchmetrics.Metric."
    return type(metric_cls.__name__, (AnomalibMetric, metric_cls), {})
