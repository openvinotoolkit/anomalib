# Feature extractor backbone

```{warning}
This section is under construction 🚧
```

This guide contains information on backbone selection for models that use pretrained backbones.

Models can use either Timm Feature Extractor or TorchFx Feature Extractor, but when selecting backbone through config, the implementation is not really important.

```{seealso}
For specifics of implementation refer to implementation classes {py:class}`Timm Feature Extractor <anomalib.models.components.feature_extractors.TimmFeatureExtractor>` and {py:class}`TorchFx Feature Extractor <anomalib.models.components.feature_extractors.TorchFXFeatureExtractor>`
```

## Backbone config

Models that support different backbones offer an option to specifiy the exact version used through parameters or config.
In the following example config, we can see that we need to specify two parameters: `backbone` and `layers` list.

```{code-block} yaml
:lineno-start: 1
:emphasize-lines: 4, 5, 6, 9

model:
  class_path: anomalib.models.Padim
  init_args:
    layers:
      - layer1
      - layer2
      - layer3
    input_size: null
    backbone: resnet18
    pre_trained: true
    n_features: null
```

## Available backbones and layers

List timm and torchfx layers and how to find them (layer4 is same as layer4.something AKA last out...). 