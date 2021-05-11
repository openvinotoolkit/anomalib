import math

import torch
from anomalib.utils.dynamic_module import DynamicBufferModule


class GaussianKDE(DynamicBufferModule):
    def __init__(self, dataset=None):
        super().__init__()

        if dataset is not None:
            self.fit(dataset)

        self.register_buffer('bw_transform', torch.Tensor())
        self.register_buffer('dataset', torch.Tensor())
        self.register_buffer('norm', torch.Tensor())

    def forward(self, features):
        # features = features.to(self.bw_transform.device)
        features = torch.matmul(features, self.bw_transform)

        estimate = torch.zeros(features.shape[0])
        for j in range(features.shape[0]):
            x = torch.sum((self.dataset - features[j]) ** 2, axis=1)
            x = torch.exp(-x / 2) * self.norm
            estimate[j] = torch.mean(x)

        return estimate

    def fit(self, dataset):
        n, d = dataset.shape

        # compute scott's bandwidth factor
        factor = n ** (-1 / (d + 4))

        cov_mat = self.cov(dataset.T, bias=False)
        inv_cov_mat = torch.linalg.inv(cov_mat)
        inv_cov = inv_cov_mat / factor ** 2

        # transform data to account for bandwidth
        bw_transform = torch.linalg.cholesky(inv_cov)
        dataset = torch.matmul(dataset, bw_transform)

        #
        norm = torch.prod(torch.diag(bw_transform))
        norm *= math.pow((2 * math.pi), (- d / 2))

        # self.register_buffer('bw_transform', bw_transform)
        self.bw_transform = bw_transform
        self.dataset = dataset
        self.norm = norm

    @staticmethod
    def cov(X, bias=False):
        mean = torch.mean(X, dim=1)
        X -= mean[:, None]
        cov = torch.matmul(X, X.T) / (X.size(1) - int(not bias))
        return cov
