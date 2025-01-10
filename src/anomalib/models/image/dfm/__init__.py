"""Deep Feature Matching (DFM) model for anomaly detection.

The DFM model extracts deep features from images using a pre-trained CNN backbone
and matches these features against a memory bank of normal samples to detect
anomalies. During inference, samples with high feature matching distances are
flagged as anomalous.

Example:
    >>> from anomalib.models.image import Dfm
    >>> model = Dfm()

The model can be used with any of the supported datasets and task modes in
anomalib.

Notes:
    The model implementation is available in the ``lightning_model`` module.

See Also:
    :class:`anomalib.models.image.dfm.lightning_model.Dfm`:
        Lightning implementation of the DFM model.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import Dfm

__all__ = ["Dfm"]
