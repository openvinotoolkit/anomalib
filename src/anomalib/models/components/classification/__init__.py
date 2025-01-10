"""Classification modules for anomaly detection.

This module provides classification components used in anomaly detection models.

Classes:
    KDEClassifier: Kernel Density Estimation based classifier for anomaly
        detection.
    FeatureScalingMethod: Enum class defining feature scaling methods for
        KDE classifier.

Example:
    >>> from anomalib.models.components.classification import KDEClassifier
    >>> from anomalib.models.components.classification import FeatureScalingMethod
    >>> # Create KDE classifier with min-max scaling
    >>> classifier = KDEClassifier(
    ...     scaling_method=FeatureScalingMethod.MIN_MAX
    ... )
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .kde_classifier import FeatureScalingMethod, KDEClassifier

__all__ = ["KDEClassifier", "FeatureScalingMethod"]
