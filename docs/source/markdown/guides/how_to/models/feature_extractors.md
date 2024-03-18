# Feature extractors

This guide demonstrates how different backbones can be used as feature extractors for anomaly detection models. Most of these models use Timm Feature Extractor except **CSFLOW** which uses TorchFx Feature Extractor. Here we show how to use API and CLI to use different backbones as feature extractors.

```{seealso}
For specifics of implementation refer to implementation classes {py:class}`Timm Feature Extractor <anomalib.models.components.feature_extractors.TimmFeatureExtractor>` and {py:class}`TorchFx Feature Extractor <anomalib.models.components.feature_extractors.TorchFXFeatureExtractor>`
```

## Available backbones and layers

::::{tab-set}

:::{tab-item} Timm

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

:::

:::{tab-item} TorchFX

When using TorchFX for feature extraction, you can use either model name, custom model, or instance of model.
In this guide, we will cover pretrained models from Torchvision passed by name. For use of the custom model or instance of a model refer to {py:class}`TorchFxFeatureExtractor class examples <anomalib.models.components.feature_extractors.TorchFXFeatureExtractor>`.

Available torchvision models are listed on [Torchvision models page](https://pytorch.org/vision/stable/models.html).

We can get layer names for selected model using the following code:

```{code-block} python
# Import model and function to list names
from torchvision.models import resnet18
from torchvision.models.feature_extraction import get_graph_node_names

# Make an instance of model with default (latest) weights
model = resnet18(weights="DEFAULT")

# Get and print node (layer) names
train_nodes, eval_nodes = get_graph_node_names(model)
print(eval_nodes)
>>>['x', 'conv1', 'bn1', 'relu', 'maxpool', ..., 'layer4.1.relu_1', 'avgpool', 'flatten', 'fc']
```

As a result, we get a list of all model nodes, which is quite long.

Now for example, if we want only output from the last node in the block named `layer4`, we specify `layer4.1.relu_1`.
If we want to avoid writing `layer4.1.relu_1` to get the last output of `layer4` block, we can shorten it to `layer4`.

We can then use selected model name and layer names with either API or using config file.

```{seealso}
Additional info about TorchFX feature extraction can be found on [PyTorch FX page](https://pytorch.org/blog/FX-feature-extraction-torchvision/) and [feature_extraction documentation page](https://pytorch.org/vision/stable/feature_extraction.html).
```

:::

::::

```{warning}
Some models might not support every backbone.
```

## Backbone and layer selection

::::{tab-set}

:::{tab-item} API

When using API, we need to specify `backbone` and `layers` when instantiating the model with a non-default backbone.

```{code-block} python
:lineno-start: 1
:emphasize-lines: 9
# Import the required modules
from anomalib.data import MVTec
from anomalib.models import Padim
from anomalib.engine import Engine

# Initialize the datamodule, model, and engine
datamodule = MVTec(num_workers=0)
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
