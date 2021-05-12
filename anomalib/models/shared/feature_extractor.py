from typing import Iterable, Callable, Dict

import torch
from torch import nn, Tensor


class FeatureExtractor(nn.Module):
    """
    FeatureExtractor [summary]

    :param nn: [description]
    :type nn: [type]

    :Example:

    >>> import torch
    >>> import torchvision
    >>> from model import FeatureExtractor

    >>> model = FeatureExtractor(model=torchvision.models.resnet18(), layers=['layer1', 'layer2', 'layer3'])
    >>> input = torch.rand((32, 3, 256, 256))
    >>> features = model(input)

    >>> [layer for layer in features.keys()]
        ['layer1', 'layer2', 'layer3']
    >>> [feature.shape for feature in features.values()]
        [torch.Size([32, 64, 64, 64]), torch.Size([32, 128, 32, 32]), torch.Size([32, 256, 16, 16])]
    """

    def __init__(self, backbone: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in self.layers}

        for layer_id in layers:
            layer = dict([*self.backbone.named_modules()])[layer_id]
            layer.register_forward_hook(self.get_features(layer_id))

    def get_features(self, layer_id: str) -> Callable:
        def hook(_, __, output):
            self._features[layer_id] = output

        return hook

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        self._features = {layer: torch.empty(0) for layer in self.layers}
        _ = self.backbone(x)
        return self._features
