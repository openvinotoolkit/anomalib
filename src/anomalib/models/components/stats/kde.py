"""Gaussian Kernel Density Estimation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch
from torch import Tensor

from anomalib.models.components.base import DynamicBufferModule


class GaussianKDE(DynamicBufferModule):
    """Gaussian Kernel Density Estimation.

    Args:
        dataset (Tensor | None, optional): Dataset on which to fit the KDE model. Defaults to None.
    """

    def __init__(self, dataset: Tensor | None = None):
        super().__init__()

        if dataset is not None:
            self.fit(dataset)

        self.register_buffer("bw_transform", Tensor())
        self.register_buffer("dataset", Tensor())
        self.register_buffer("norm", Tensor())

        self.bw_transform = Tensor()
        self.dataset = Tensor()
        self.norm = Tensor()

    def forward(self, features: Tensor) -> Tensor:
        """Get the KDE estimates from the feature map.

        Args:
          features (Tensor): Feature map extracted from the CNN

        Returns: KDE Estimates
        """
        features = torch.matmul(features, self.bw_transform)

        estimate = torch.zeros(features.shape[0]).to(features.device)
        for i in range(features.shape[0]):
            embedding = ((self.dataset - features[i]) ** 2).sum(dim=1)
            embedding = torch.exp(-embedding / 2) * self.norm
            estimate[i] = torch.mean(embedding)

        return estimate

    def fit(self, dataset: Tensor) -> None:
        """Fit a KDE model to the input dataset.

        Args:
          dataset (Tensor): Input dataset.

        Returns:
            None
        """
        num_samples, dimension = dataset.shape

        # compute scott's bandwidth factor
        factor = num_samples ** (-1 / (dimension + 4))

        cov_mat = self.cov(dataset.T)
        inv_cov_mat = torch.linalg.inv(cov_mat)
        inv_cov = inv_cov_mat / factor**2

        # transform data to account for bandwidth
        bw_transform = torch.linalg.cholesky(inv_cov)
        dataset = torch.matmul(dataset, bw_transform)

        #
        norm = torch.prod(torch.diag(bw_transform))
        norm *= math.pow((2 * math.pi), (-dimension / 2))

        self.bw_transform = bw_transform
        self.dataset = dataset
        self.norm = norm

    @staticmethod
    def cov(tensor: Tensor) -> Tensor:
        """Calculate the unbiased covariance matrix.

        Args:
            tensor (Tensor): Input tensor from which covariance matrix is computed.

        Returns:
            Output covariance matrix.
        """
        mean = torch.mean(tensor, dim=1)
        tensor -= mean[:, None]
        cov = torch.matmul(tensor, tensor.T) / (tensor.size(1) - 1)
        return cov
