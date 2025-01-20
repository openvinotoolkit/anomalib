"""Deep Spatial Reconstruction (DSR) model.

DSR is an anomaly detection model that uses a deep autoencoder architecture to
learn spatial reconstructions of normal images. The model learns to reconstruct
normal patterns and identifies anomalies based on reconstruction errors.

Example:
    >>> from anomalib.models.image import Dsr
    >>> model = Dsr()

The model can be used with any of the supported datasets and task modes in
anomalib.

Notes:
    The model implementation is available in the ``lightning_model`` module.

See Also:
    :class:`anomalib.models.image.dsr.lightning_model.Dsr`:
        Lightning implementation of the DSR model.
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import Dsr

__all__ = ["Dsr"]
