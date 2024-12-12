"""Base Visualizer."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from lightning.pytorch import Callback


class Visualizer(Callback):
    """Base class for all visualizers.

    In Anomalib, the visualizer is used to visualize the results of the model
    during the testing and prediction phases.
    """
