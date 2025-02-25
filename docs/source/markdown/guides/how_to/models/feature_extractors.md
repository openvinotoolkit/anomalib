# Feature extractor

This guide demonstrates how different backbones can be used as feature extractors for anomaly detection models. Here we show how to use API and CLI to use different backbones as feature extractors.

Backbones can be defined by their name as available in [Timm](https://github.com/huggingface/pytorch-image-models#models)
or directly passed to the feature extractor as torch `nn.Module`.

```{seealso}
For specifics of implementation refer to implementation classes {py:class}`Timm Feature Extractor <anomalib.models.components.feature_extractors.TimmFeatureExtractor>`
```

## Available backbones and layers

Available Timm models are listed on [Timm GitHub page](https://github.com/huggingface/pytorch-image-models#models).

In most cases, we want to use a pretrained backbone, so can get a list of all such models using the following code:

```{code-block} python
import timm
# list all pretrained models in timm
for model_name in timm.list_models(pretrained=True):
    print(model_name)
```

Once we have a model selected we can obtain available layer names using the following code:

```{code-block} python
import timm
model = timm.create_model("resnet18", features_only=True)
# Print module names
print(model.feature_info.module_name())
>>>['act1', 'layer1', 'layer2', 'layer3', 'layer4']

model = timm.create_model("mobilenetv3_large_100", features_only=True)
print(model.feature_info.module_name())
>>>['blocks.0.0', 'blocks.1.1', 'blocks.2.2', 'blocks.4.1', 'blocks.6.0']
```

We can then use selected model name and layer names with either API or using config file.

```{warning}
Some models might not support every backbone.
```

## Backbone and layer selection

::::{tab-set}

:::{tab-item} API

When using API, we need to specify `backbone` and `layers` when instantiating the model with a non-default timm backbone.

```{code-block} python
:lineno-start: 1
:emphasize-lines: 9
# Import the required modules
from anomalib.data import MVTecAD
from anomalib.models import Padim
from anomalib.engine import Engine

# Initialize the datamodule, model, and engine
datamodule = MVTecAD(num_workers=0)
# Specify backbone and layers
model = Padim(backbone="resnet18", layers=["layer1", "layer2"])
engine = Engine(image_metrics=["AUROC"], pixel_metrics=["AUROC"])

# Train the model
engine.fit(datamodule=datamodule, model=model)
```

:::

:::{tab-item} CLI

In the following example config, we can see that we need to specify two parameters: the `backbone` and `layers` list.

```{code-block} yaml
:lineno-start: 1
:emphasize-lines: 4, 5, 6, 8

model:
  class_path: anomalib.models.Padim
  init_args:
    layers:
      - blocks.1.1
      - blocks.2.2
    input_size: null
    backbone: mobilenetv3_large_100
    pre_trained: true
    n_features: 50
```

Then we can train using:

```{code-block} bash
anomalib train --config <path/to/config>
```

:::

::::

## Custom backbone

To use a custom backbone model, e.g. locally saved models or models with modified weights, the model instance
is passed as backbone argument. Features will be extracted from the specified layers. Make sure that the backbone model
actually possesses the layers from which the feature will be extracted.

```{code-block} python
:lineno-start: 1
# Import the required modules
import torch
from torchvision import models

from anomalib.data import MVTec
from anomalib.models import Padim
from anomalib.engine import Engine

# Initialize the datamodule, model, and engine
datamodule = MVTec(num_workers=0)

# Specify custom model
weights = torch.hub.load_state_dict_from_url("https://huggingface.co/mzweilin/robust-imagenet-models/resolve/main/wide_resnet50_2_l2_eps5.pth")
custom_backbone = models.wide_resnet50_2()
custom_backbone.load_state_dict(weights)

# Specify backbone and layers
model = Padim(backbone=custom_backbone, layers=["layer1", "layer3"])
engine = Engine(image_metrics=["AUROC"], pixel_metrics=["AUROC"])

# Train the model
engine.fit(datamodule=datamodule, model=model)
```
