"""Deep Feature Kernel Density Estimation (DFKDE) model for anomaly detection.

The DFKDE model extracts deep features from images using a pre-trained CNN backbone
and fits a kernel density estimation on these features to model the distribution
of normal samples. During inference, samples with low likelihood under this
distribution are flagged as anomalous.

Example:
    >>> from anomalib.models.image import Dfkde
    >>> model = Dfkde()

The model can be used with any of the supported datasets and task modes in
anomalib.

Notes:
    The model implementation is available in the ``lightning_model`` module.

See Also:
    :class:`anomalib.models.image.dfkde.lightning_model.Dfkde`:
        Lightning implementation of the DFKDE model.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import Dfkde

__all__ = ["Dfkde"]
