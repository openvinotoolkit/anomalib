"""PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.

The PaDiM model is an anomaly detection approach that leverages patch-based
distribution modeling using pretrained CNN feature embeddings. It models the
distribution of patch embeddings at each spatial location using multivariate
Gaussian distributions.

The model uses features extracted from multiple layers of networks like
``ResNet`` to capture both semantic and low-level visual information. During
inference, it computes Mahalanobis distances between test patch embeddings and
their corresponding reference distributions to detect anomalies.

Paper: https://arxiv.org/abs/2011.08785
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import Padim

__all__ = ["Padim"]
