import torch

from anomalib.models.shared.dynamic_module import DynamicBufferModule


class PCA(DynamicBufferModule):
    def __init__(self, n_components: int):
        super().__init__()
        self.n_components = n_components

        self.register_buffer("U", torch.Tensor())
        self.register_buffer("S", torch.Tensor())
        self.register_buffer("V", torch.Tensor())
        self.register_buffer("mean", torch.Tensor())

    def fit_transform(self, dataset: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(dataset, axis=0)
        dataset -= mean

        U, S, V = torch.svd(dataset)

        self.U = U
        self.S = S
        self.V = V
        self.mean = mean

        return torch.matmul(dataset, V[:, : self.n_components])

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        y -= self.mean
        return torch.matmul(y, self.V[:, : self.n_components])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)
