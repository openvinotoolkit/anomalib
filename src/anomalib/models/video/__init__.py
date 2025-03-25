"""Anomalib Video Models.

This module contains implementations of various deep learning models for video-based
anomaly detection.

Example:
    >>> from anomalib.models.video import AiVad
    >>> from anomalib.data import Avenue
    >>> from anomalib.engine import Engine

    >>> # Initialize a model and datamodule
    >>> datamodule = Avenue(
    ...     clip_length_in_frames=2,
    ...     frames_between_clips=1,
    ...     target_frame=VideoTargetFrame.LAST
    ... )
    >>> model = AiVad()

    >>> # Train using the engine
    >>> engine = Engine()  # doctest: +SKIP
    >>> engine.fit(model=model, datamodule=datamodule)  # doctest: +SKIP

    >>> # Get predictions
    >>> predictions = engine.predict(model=model, datamodule=datamodule)  # doctest: +SKIP

Available Models:
    - :class:`AiVad`: AI-based Video Anomaly Detection
"""

# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .ai_vad import AiVad
from .fuvas import Fuvas

__all__ = ["AiVad", "Fuvas"]
