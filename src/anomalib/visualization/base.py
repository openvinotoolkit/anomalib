"""Base visualization module for anomaly detection.

This module provides the base ``Visualizer`` class that defines the interface for
visualizing anomaly detection results. The key components include:

    - Base ``Visualizer`` class that inherits from PyTorch Lightning's ``Callback``
    - Interface for visualizing model outputs during testing and prediction
    - Support for customizable visualization formats and configurations

Example:
    >>> from anomalib.visualization import Visualizer
    >>> # Create custom visualizer
    >>> class CustomVisualizer(Visualizer):
    ...     def visualize(self, **kwargs):
    ...         # Custom visualization logic
    ...         pass

The module ensures consistent visualization by:
    - Providing a standardized visualization interface
    - Supporting both classification and segmentation results
    - Enabling customizable visualization formats
    - Maintaining consistent output formats

Note:
    All visualizer implementations should inherit from the base ``Visualizer``
    class and implement the required visualization methods.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from lightning.pytorch import Callback


class Visualizer(Callback):
    """Base class for all visualizers.

    This class serves as the foundation for implementing visualization functionality in
    Anomalib. It inherits from PyTorch Lightning's ``Callback`` class to integrate with
    the training workflow.

    The visualizer is responsible for generating visual representations of model outputs
    during testing and prediction phases. This includes:

        - Visualizing input images
        - Displaying model predictions
        - Showing ground truth annotations
        - Generating overlays and heatmaps
        - Saving visualization results

    Example:
        >>> from anomalib.visualization import Visualizer
        >>> # Create custom visualizer
        >>> class CustomVisualizer(Visualizer):
        ...     def visualize(self, **kwargs):
        ...         # Custom visualization logic
        ...         pass

    Note:
        All custom visualizers should:
            - Inherit from this base class
            - Implement the ``visualize`` method
            - Handle relevant visualization configurations
            - Maintain consistent output formats
    """
