"""GANomaly Algorithm Implementation.

GANomaly is an anomaly detection model that uses a conditional GAN architecture to
learn the normal data distribution. The model consists of a generator network that
learns to reconstruct normal images, and a discriminator that helps ensure the
reconstructions are realistic.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Ganomaly
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = Ganomaly()
    >>> engine = Engine()

    >>> engine.fit(model, datamodule=datamodule)  # doctest: +SKIP
    >>> predictions = engine.predict(model, datamodule=datamodule)  # doctest: +SKIP

Paper:
    Title: GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training
    URL: https://arxiv.org/abs/1805.06725

See Also:
    :class:`anomalib.models.image.ganomaly.lightning_model.Ganomaly`:
        PyTorch Lightning implementation of the GANomaly model.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import Ganomaly

__all__ = ["Ganomaly"]
