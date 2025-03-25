"""FUVAS: Few-shot Unsupervised Video Anomaly Segmentation via Low-Rank Factorization of Spatio-Temporal Features.

The FUVAS model extracts deep features from video clips using a pre-trained 3D CNN/transformer
backbone and fits a PCA-based reconstruction model to detect anomalies. The model computes
feature reconstruction errors to identify anomalous frames and regions in videos.

Example:
    >>> from anomalib.models.video import Fuvas
    >>> model = Fuvas(
    ...     backbone="x3d_s",
    ...     layer="blocks.4"
    ... )

The model can be used with video anomaly detection datasets supported in anomalib.

Notes:
    The model implementation is available in the ``lightning_model`` module.

See Also:
    :class:`anomalib.models.video.fuvas.lightning_model.Fuvas`:
        Lightning implementation of the FUVAS model.
"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import Fuvas

__all__ = ["Fuvas"]
