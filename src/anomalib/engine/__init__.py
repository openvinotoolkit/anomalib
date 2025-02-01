"""Engine module for training and evaluating anomaly detection models.

This module provides functionality for training and evaluating anomaly detection
models. The main component is the :class:`Engine` class which handles:

- Model training and validation
- Metrics computation and logging
- Checkpointing and model export
- Distributed training support

Example:
    Create and use an engine:

    >>> from anomalib.engine import Engine
    >>> engine = Engine()
    >>> engine.train()  # doctest: +SKIP
    >>> engine.test()  # doctest: +SKIP

    The engine can also be used with a custom configuration:

    >>> from anomalib.config import Config
    >>> config = Config(path="config.yaml")
    >>> engine = Engine(config=config)  # doctest: +SKIP
"""

# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .accelerator import XPUAccelerator
from .engine import Engine
from .strategy import SingleXPUStrategy

__all__ = ["Engine", "SingleXPUStrategy", "XPUAccelerator"]
