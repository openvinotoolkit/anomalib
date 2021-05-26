from typing import List
from typing import Optional

import torch
from torch import Tensor


class MultiVariateGaussian(torch.nn.Module):
    @staticmethod
    def _cov(x: Tensor, rowvar: bool = False, bias: bool = False, ddof: Optional[int] = None, aweights=None) -> Tensor:
        """
        Estimates covariance matrix like numpy.cov
        :param x: A 1-D or 2-D array containing multiple variables and observations.
                        Each row of `m` represents a variable, and each column a single
                        observation of all those variables. Also see `rowvar` below.
        :param rowvar: If `rowvar` is True (default), then each row represents a
                        variable, with observations in the columns. Otherwise, the relationship
                        is transposed: each column represents a variable, while the rows
                        contain observations.
        :param bias: Default normalization (False) is by ``(N - 1)``, where ``N`` is the
                        number of observations given (unbiased estimate). If `bias` is True,
                        then normalization is by ``N``. These values can be overridden by using
                        the keyword ``ddof`` in numpy versions >= 1.5.
        :param ddof: If not ``None`` the default value implied by `bias` is overridden.
                        Note that ``ddof=1`` will return the unbiased estimate, even if both
                        `fweights` and `aweights` are specified, and ``ddof=0`` will return
                        the simple average. See the notes for the details. The default value
                        is ``None``.
        :param aweights: 1-D array of observation vector weights. These relative weights are
                        typically large for observations considered "important" and smaller for
                        observations considered less "important". If ``ddof=0`` the array of
                        weights can be used to assign probabilities to observation vectors.
        :return: The covariance matrix of the variables.
        """
        # ensure at least 2D
        if x.dim() == 1:
            x = x.view(-1, 1)

        # treat each column as a data point, each row as a variable
        if rowvar and x.shape[0] != 1:
            x = x.t()

        if ddof is None:
            if bias == 0:
                ddof = 1
            else:
                ddof = 0

        w = aweights
        if w is not None:
            if not torch.is_tensor(w):
                w = torch.tensor(w, dtype=torch.float)
            w_sum = torch.sum(w)
            avg = torch.sum(x * (w / w_sum)[:, None], 0)
        else:
            avg = torch.mean(x, 0)

        # Determine the normalization
        if w is None:
            fact = x.shape[0] - ddof
        elif ddof == 0:
            fact = w_sum
        elif aweights is None:
            fact = w_sum - ddof
        else:
            fact = w_sum - ddof * torch.sum(w * w) / w_sum

        xm = x.sub(avg.expand_as(x))

        if w is None:
            X_T = xm.t()
        else:
            X_T = torch.mm(torch.diag(w), xm).t()

        c = torch.mm(X_T, xm)
        c = c / fact

        return c.squeeze()

    def forward(self, embedding: Tensor) -> List[Tensor]:
        """
        Calculate multivariate Gaussian distribution
        :param embedding: CNN features whose dimensionality is reduced via either random sampling or PCA.
        :return: mean and covariance of the multi-variate gaussian distribution that fits the features.
        """
        device = embedding.device

        batch, channel, height, width = embedding.size()
        embedding_vectors = embedding.view(batch, channel, height * width)
        mean = torch.mean(embedding_vectors, dim=0)
        cov = torch.zeros(size=(channel, channel, height * width), device=device)
        identity = torch.eye(channel).to(device)
        for i in range(height * width):
            cov[:, :, i] = self._cov(embedding_vectors[:, :, i], rowvar=False) + 0.01 * identity

        return [mean, cov]

    def fit(self, embedding: Tensor) -> List[Tensor]:
        return self.forward(embedding)
