import torch.nn.functional as F
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
import torch

class FeatureExtractor(nn.Module):
    def __init__(self, model, layer_name, pool_factor=None):
        super(FeatureExtractor, self).__init__()
        return_nodes = [layer_name]
        self.model = create_feature_extractor(model, return_nodes=return_nodes)

        self.layer_name = layer_name
        self.pool_factor = pool_factor
        self.feature_shapes = dict()
        self.feature_shapes_flatten = dict()

    def __repr__(self):
        out = 'Layer {}\n'.format(self.layer_name)
        if self.pool_factor:
            out = '{}Pool factors {}\n'.format(out, list(self.pool_factor.values()))
        out = '{}'.format(self.model.__repr__())
        return out

    def forward(self, data):
        # pool = self.pool_factor
        out = self.model(data)
        return out

    def get_feature_shapes(self):
        return self.feature_shapes[self.layer_name]
    
    def get_flat_shapes(self):
        return self.feature_shapes_flatten[self.layer_name]