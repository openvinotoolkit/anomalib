"""Multi Variate Gaussian Distribution."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Any, List, Optional

import torch
from torch import Tensor, nn


class MultiVariateGaussian(nn.Module):
    """Multi Variate Gaussian Distribution."""

    def __init__(self, n_features, n_patches):
        super().__init__()

        self.register_buffer("mean", torch.zeros(n_features, n_patches))
        self.register_buffer("inv_covariance", torch.eye(n_features).unsqueeze(0).repeat(n_patches, 1, 1))

        self.mean: Tensor
        self.inv_covariance: Tensor

    @staticmethod
    def _cov(
        observations: Tensor,
        rowvar: bool = False,
        bias: bool = False,
        ddof: Optional[int] = None,
        aweights: Tensor = None,
    ) -> Tensor:
        """Estimates covariance matrix like numpy.cov.

        Args:
            observations (Tensor): A 1-D or 2-D array containing multiple variables and observations.
                 Each row of `m` represents a variable, and each column a single
                 observation of all those variables. Also see `rowvar` below.
            rowvar (bool): If `rowvar` is True (default), then each row represents a
                variable, with observations in the columns. Otherwise, the relationship
                is transposed: each column represents a variable, while the rows
                contain observations. Defaults to False.
            bias (bool): Default normalization (False) is by ``(N - 1)``, where ``N`` is the
                number of observations given (unbiased estimate). If `bias` is True,
                then normalization is by ``N``. These values can be overridden by using
                the keyword ``ddof`` in numpy versions >= 1.5. Defaults to False
            ddof (Optional, int): If not ``None`` the default value implied by `bias` is overridden.
                Note that ``ddof=1`` will return the unbiased estimate, even if both
                `fweights` and `aweights` are specified, and ``ddof=0`` will return
                the simple average. See the notes for the details. The default value
                is ``None``.
            aweights (Tensor): 1-D array of observation vector weights. These relative weights are
                typically large for observations considered "important" and smaller for
                observations considered less "important". If ``ddof=0`` the array of
                weights can be used to assign probabilities to observation vectors. (Default value = None)


        Returns:
          The covariance matrix of the variables.
        """
        # ensure at least 2D
        if observations.dim() == 1:
            observations = observations.view(-1, 1)

        # treat each column as a data point, each row as a variable
        if rowvar and observations.shape[0] != 1:
            observations = observations.t()

        if ddof is None:
            if bias == 0:
                ddof = 1
            else:
                ddof = 0

        weights = aweights
        weights_sum: Any

        if weights is not None:
            if not torch.is_tensor(weights):
                weights = torch.tensor(weights, dtype=torch.float)  # pylint: disable=not-callable
            weights_sum = torch.sum(weights)
            avg = torch.sum(observations * (weights / weights_sum)[:, None], 0)
        else:
            avg = torch.mean(observations, 0)

        # Determine the normalization
        if weights is None:
            fact = observations.shape[0] - ddof
        elif ddof == 0:
            fact = weights_sum
        elif aweights is None:
            fact = weights_sum - ddof
        else:
            fact = weights_sum - ddof * torch.sum(weights * weights) / weights_sum

        observations_m = observations.sub(avg.expand_as(observations))

        if weights is None:
            x_transposed = observations_m.t()
        else:
            x_transposed = torch.mm(torch.diag(weights), observations_m).t()

        covariance = torch.mm(x_transposed, observations_m)
        covariance = covariance / fact

        return covariance.squeeze()

    def forward(self, embedding: Tensor) -> List[Tensor]:
        """Calculate multivariate Gaussian distribution.

        Args:
          embedding (Tensor): CNN features whose dimensionality is reduced via either random sampling or PCA.

        Returns:
          mean and inverse covariance of the multi-variate gaussian distribution that fits the features.
        """
        device = embedding.device

        batch, channel, height, width = embedding.size()
        embedding_vectors = embedding.view(batch, channel, height * width)
        self.mean = torch.mean(embedding_vectors, dim=0)
        covariance = torch.zeros(size=(channel, channel, height * width), device=device)
        identity = torch.eye(channel).to(device)
        for i in range(height * width):
            covariance[:, :, i] = self._cov(embedding_vectors[:, :, i], rowvar=False) + 0.01 * identity

        # calculate inverse covariance as we need only the inverse
        self.inv_covariance = torch.linalg.inv(covariance.permute(2, 0, 1))

        return [self.mean, self.inv_covariance]

    def fit(self, embedding: Tensor) -> List[Tensor]:
        """Fit multi-variate gaussian distribution to the input embedding.

        Args:
            embedding (Tensor): Embedding vector extracted from CNN.

        Returns:
            Mean and the covariance of the embedding.
        """
        return self.forward(embedding)
