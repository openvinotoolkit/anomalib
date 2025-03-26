import torch
from torch import nn
import copy


class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, train_backbone=False):
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.train_backbone = train_backbone
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            self.register_hook(extract_layer)

        self.to(self.device)

    def forward(self, images, eval=True):
        self.outputs.clear()
        if self.train_backbone and not eval:
            self.backbone(images)
        else:
            with torch.no_grad():
                try:
                    _ = self.backbone(images)
                except LastLayerToExtractReachedException:
                    pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]

    def register_hook(self, layer_name):
        module = self.find_module(self.backbone, layer_name)
        if module is not None:
            forward_hook = ForwardHook(
                self.outputs, layer_name, self.layers_to_extract_from[-1]
            )
            if isinstance(module, torch.nn.Sequential):
                hook = module[-1].register_forward_hook(forward_hook)
            else:
                hook = module.register_forward_hook(forward_hook)
            self.backbone.hook_handles.append(hook)
        else:
            raise ValueError(f"Module {layer_name} not found in the model")

    def find_module(self, model, module_name):
        for name, module in model.named_modules():
            if name == module_name:
                return module
            elif "." in module_name:
                father, child = module_name.split(".", 1)
                if name == father:
                    return self.find_module(module, child)
        return None


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        return None


class LastLayerToExtractReachedException(Exception):
    pass
