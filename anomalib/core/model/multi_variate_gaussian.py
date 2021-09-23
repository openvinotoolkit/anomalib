"""
Multi Variate Gaussian Distribution
"""

from typing import Any, List, Optional

import torch
from torch import Tensor

from anomalib.core.model.dynamic_module import DynamicBufferModule


class MultiVariateGaussian(DynamicBufferModule):
    """
    Multi Variate Gaussian Distribution
    """

    def __init__(self):
        super().__init__()

        self.register_buffer("mean", torch.Tensor())
        self.register_buffer("covariance", torch.Tensor())

        self.mean: Optional[Tensor] = None
        self.covariance: Optional[Tensor] = None

    @staticmethod
    def _cov(
        observations: Tensor, rowvar: bool = False, bias: bool = False, ddof: Optional[int] = None, aweights=None
    ) -> Tensor:
        """Estimates covariance matrix like numpy.cov

        Args:
            observations: A 1-D or 2-D array containing multiple variables and observations.
                 Each row of `m` represents a variable, and each column a single
                 observation of all those variables. Also see `rowvar` below.
            rowvar: If `rowvar` is True (default), then each row represents a
                variable, with observations in the columns. Otherwise, the relationship
                is transposed: each column represents a variable, while the rows
                contain observations.
            bias: Default normalization (False) is by ``(N - 1)``, where ``N`` is the
                number of observations given (unbiased estimate). If `bias` is True,
                then normalization is by ``N``. These values can be overridden by using
                the keyword ``ddof`` in numpy versions >= 1.5.
            ddof: If not ``None`` the default value implied by `bias` is overridden.
                Note that ``ddof=1`` will return the unbiased estimate, even if both
                `fweights` and `aweights` are specified, and ``ddof=0`` will return
                the simple average. See the notes for the details. The default value
                is ``None``.
            aweights: 1-D array of observation vector weights. These relative weights are
                typically large for observations considered "important" and smaller for
                observations considered less "important". If ``ddof=0`` the array of
                weights can be used to assign probabilities to observation vectors. (Default value = None)
            x: Tensor:
            rowvar: bool:  (Default value = False)
            bias: bool:  (Default value = False)
            ddof: Optional[int]:  (Default value = None)

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
        """
        Calculate multivariate Gaussian distribution

        Args:
          embedding: CNN features whose dimensionality is reduced via either random sampling or PCA.
          embedding: Tensor:

        Returns:
          mean and covariance of the multi-variate gaussian distribution that fits the features.

        """
        device = embedding.device

        batch, channel, height, width = embedding.size()
        embedding_vectors = embedding.view(batch, channel, height * width)
        self.mean = torch.mean(embedding_vectors, dim=0)
        self.covariance = torch.zeros(size=(channel, channel, height * width), device=device)
        identity = torch.eye(channel).to(device)
        for i in range(height * width):
            self.covariance[:, :, i] = self._cov(embedding_vectors[:, :, i], rowvar=False) + 0.01 * identity

        return [self.mean, self.covariance]

    def fit(self, embedding: Tensor) -> List[Tensor]:
        """
        Fit multi-variate gaussian distribution to the input embedding.

        Args:
            embedding: Tensor: Embedding vector extracted from CNN.

        Returns:
            Mean and the covariance of the embedding.

        """
        return self.forward(embedding)
