```{eval-rst}
:orphan:
```

# Pre-processing in Anomalib

This guide explains how pre-processing works in Anomalib, its integration with models, and how to create custom pre-processors.

## Prerequisites

- {doc}`Transforms <../data/transforms>`

```{note}
Before reading this guide, it is important that you are familiar with the concept of model-specific transforms, see {doc}`Transforms <../data/transforms>`.
```

## Overview

Anomalib's pre-processing step consists of applying the model-specific input data transforms (like input size and normalization) to the images before passing the images to the model. This ensures that the images are in the format that is expected by the model. The module that handles the pre-processing steps in Anomalib is the `PreProcessor`. The `PreProcessor` is responsible for applying the transforms to the input images, and handling any stage-specific logic related to this.

Another important role of the `PreProcessor` is to encapsulate the model-specific transforms within the model graph during export. This design ensures that during deployment:

- Pre-processing is part of the exported model (ONNX, OpenVINO)
- Users don't need to manually resize or normalize inputs
- Edge deployment is simplified with automatic pre-processing

To achieve this, the `PreProcessor` class implements the following components:

1. A Lightning Callback for managing stage-specific pre-processing steps (e.g. training, validation, testing)
2. A PyTorch Module for transform application that gets exported with the model

## Basic Usage

To create a simple pre-processor, simply define some transforms and pass them to a new `PreProcessor` instance using the `transform` argument.

```python
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Normalize, Resize

transform = Compose([
    Resize((300, 300)),
    Normalize(mean=[0.43, 0.48, 0.45], std=[0.23, 0.22, 0.25]),
])
pre_processor = PreProcessor(transform=transform)
```

The newly created pre-processor is fully compatible with Anomalib. It can be used it in an Anomalib workflow by passing it to the model using the `pre_processor` argument. Let's try this with a Fastflow model.

```python
from anomalib.models import FastFlow

model = Fastflow(pre_processor=pre_processor)
```

The pre-processor which we created earlier is now attached to the Fastflow model. Let's print the transform stored in the pre-processor to confirm that the model contains our transform:

```python
print(model.pre_processor.transform)
# Compose(
#       Resize(size=[300, 300], interpolation=InterpolationMode.BILINEAR, antialias=True)
#       Normalize(mean=[0.43, 0.48, 0.45], std=[0.23, 0.22, 0.25], inplace=False)
# )
```

We can now create an engine and run an anomaly detection workflow. In each stage of the pipeline, the pre-processor will use its callback hooks to intercept the batch before it is passed to the model, and update the contents of the batch by applying the transform.

```python
from anomalib.engine import Engine

engine = Engine()
engine.train(model, datamodule=datamodule)  # pre-processor stored in the model will be used for input transforms
```

### Exporting

In addition to applying the transforms in the callback hooks, the pre-processor also applies the transforms in the forward pass of the model. As a result, the transforms will get included in the model graph when it is exported to ONNX or OpenVINO format. The transform that is used for exporting is a modified version of the original transform. This is needed because not all operations from Torchvision's standard transforms are compabitle with ONNX. The exported version of the transform can be inspected using the `export_transform` attribute of the pre-processor. The exported transform is obtained internally using a utility function that replaces several commonly used operations with a modified, exportable counterpart.

```python
from anomalib.pre_processing import PreProcessor
from anomalib.pre_processing.utils.transform import get_exportable_transform

transform = Compose([
      Resize(size=(256, 256)),
      CenterCrop(size=(224, 224)),
      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
pre_processor = PreProcessor(transform=transform)

print(pre_processor.transform)
# Compose(
#       Resize(size=[256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True)
#       CenterCrop(size=(224, 224))
#       Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
# )
print(pre_processor.export_transform)
# Compose(
#       Resize(size=[256, 256], interpolation=InterpolationMode.BILINEAR, antialias=False)
#       ExportableCenterCrop(size=[224, 224])
#       Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
# )

exportable_transform = get_exportable_transform(pre_processor.transform)
print(exportable_transform == pre_processor.export_transform)  # True
```

```{note}
The exportable transform that is infused in the model graph uses slightly different operations compared to the standard transform that is used during training and evaluation. This may cause small differences in the final model predictions between the standard model and the exported model. When encountering unexpected behaviour of your exported model, a good first step may be to confirm that the exported transforms are working as intended.
```

```{note}
The `get_exportable_transform` function supports conversion of several commonly used transforms. It may occur that your custom set of transforms contains a transform that is not compatible with ONNX but is also not supported by `get_exportable_transforms`. In this case, please feel free to submit a [Feature Request](https://github.com/openvinotoolkit/anomalib/discussions/new?category=feature-requests) in our Discussions section on Github.
```

After training a model, we can export our model to ONNX format, and our custom set of transforms automatically gets applied when running the model in onnxruntime.

```python
# Export model with pre-processing included
model.export("model.onnx")

# During deployment - no manual pre-processing needed
deployed_model = onnxruntime.InferenceSession("model.onnx")
raw_image = cv2.imread("test.jpg")  # Any size, unnormalized
prediction = deployed_model.run(None, {"input": raw_image})
```

## Default Pre-processor

The example above illustrated how to create a `PreProcessor` instance and pass it to an Anomalib model. Depending on the use-case, this may not always be necessary. When the user does not pass a `PreProcessor` instance to the model, the model will automatically configure a `PreProcessor` instance that applies a default set of model-specific transforms. Let's inspect the `pre_processor` attribute of a default Padim model:

```python
from anomalib.models import Padim

model = Padim()
print(model.pre_processor)
# PreProcessor(
#   (transform): Compose(
#         Resize(size=[256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True)
#         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
#   )
#   (export_transform): Compose(
#         Resize(size=[256, 256], interpolation=InterpolationMode.BILINEAR, antialias=False)
#         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
#   )
# )
```

As you can see, Padim has automatically configured a `PreProcessor` instance which contains a `Resize` and a `Normalize` transform as its default model-specific transforms.

Internally, the default pre-processor is configured with the `configure_pre_processor` method, which each subclass of `AnomalibModule` is expected to implement. Let's see what happens if we call Padim's implementation of this method directly.

```python
print(Padim.configure_pre_processor())
# PreProcessor(
#   (transform): Compose(
#         Resize(size=[256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True)
#         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
#   )
#   (export_transform): Compose(
#         Resize(size=[256, 256], interpolation=InterpolationMode.BILINEAR, antialias=False)
#         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
#   )
# )
```

It's the same pre-processor as earlier! The `configure_pre_processor` method can be a useful tool to inspect the default pre-processing behaviour and model-specific transforms of an Anomalib model, without first having to create a model instance. To illustrate why this is useful, consider the following example where we want to change the input normalization for a Patchcore model, but keep the other model-specific transforms unchanged. In this case, we can call `configure_pre_processor` to inspect the default set of model-specific transforms, and then create a new pre-processor with a modified `Normalize` transform.

```python
from anomalib.models import Patchcore

print(Patchcore.configure_pre_processor().transform)
# Compose(
#       Resize(size=[256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True)
#       CenterCrop(size=(224, 224))
#       Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
# )

from torchvision.transforms.v2 import Compose, CenterCrop, Normalize, Resize

# replace the Normalize transform, but replicate the other transforms
transform = Compose([
    Resize(size=[256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True),
    CenterCrop(size=(224, 224)),
    Normalize([0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),  # CLIP stats
])
pre_processor = PreProcessor(transform=transform)
```

The `configure_pre_processor` method contains a useful shortcut for updating the image size (which is the most common use-case for custom transforms). Passing a size tuple to the `image_size` argument of the `configure_pre_processor` method yields a pre-processor with an updated `Resize` transform, as shown below:

```python
from anomalib.models import Padim
pre_processor = Padim.configure_pre_processor(image_size=(200, 200))

print(pre_processor.transform)
# Compose(
#       Resize(size=[200, 200], interpolation=InterpolationMode.BILINEAR, antialias=True)
#       Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
# )

model = Padim(pre_processor=pre_processor)
```

Finally, in some cases it may be desired to disable pre-processing entirely. This is done by passing `False` to the model's pre_processor argument.

```python
model = Padim(pre_processor=False)
print(model.pre_processor is None)  # True
```

Note that it is rarely recommended to disable pre-processing in Anomalib workflows. This functionality is intended for advanced use-cases or demo purposes.

## Custom Pre-processor Class

Advanced users may want to define their own pre-processing pipeline. This can be useful when additional logic is needed, or when the pre-processing behaviour should differ between the stages of the Anomalib workflow. As an example, let's create a PreProcessor which uses separate transforms for each of the train, val and test stages:

```python
from anomalib.pre_processing import PreProcessor
from anomalib.utils.transform import get_exportable_transform
from torchvision.transforms.v2 import Transform


class StageSpecificPreProcessor(PreProcessor):

    def __init__(
        self,
        train_transform: Transform | None = None,
        val_transform: Transform | None = None,
        test_transform: Transform | None = None,
    ):
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.export_transform = get_exportable_transform(test_transform)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.train_transform:
            batch.image, batch.gt_mask = self.train_transform(batch.image, batch.gt_mask)

    def on_val_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.val_transform:
            batch.image, batch.gt_mask = self.val_transform(batch.image, batch.gt_mask)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.test_transform:
            batch.image, batch.gt_mask = self.test_transform(batch.image, batch.gt_mask)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.test_transform:
            batch.image, batch.gt_mask = self.test_transform(batch.image, batch.gt_mask)
```

Now that we have defined a custom `PreProcessor` sublass, we can create an instance and pass some transforms for the different stages. Just like the standard `PreProcessor`, we can add the new `PreProcessor` to any Anomalib model to use its stage-specific transforms in the Anomalib workflow:

```python
from torchvision.transforms.v2 import Compose, Centercrop, RandomCrop, Resize

train_transform = Resize((224, 224))
val_transform = Compose([
    Resize((256, 256)),
    CenterCrop((224, 224))
])
test_transform = Compose([
    Resize((256, 256)),
    RandomCrop((224, 224)),

])

pre_processor = StageSpecificPreProcessor(
    train_transform=train_transform,
    val_transform=val_transform,
    test_transform=test_transform,
)
# add the custom pre-processor to an Anomalib model.
model = MyModel(pre_processor=pre_processor)
```

```{note}
The example above is for illustrative purposes only. In practice, it would rarely be sensible to use different model-specific transforms for different stages. This should not be confused with **data augmentations**, where different augmentation transforms for different stages is a valid use-case. For further reading about the differences between model-specific transforms and data augmentations, please refer to our {doc}`Transforms guide <../data/transforms>`.
```

## Best Practices

## Common Pitfalls

### 1. Omitting required model-specific transforms

In many cases we only want to change a specific part of the model-specific transforms, such as the input size. We need to be careful that we don't omit any other model-specific transforms that the model may need

```python
from anomalib.models import Padim
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Normalize, Resize

# Wrong: only specify the new resize, without considering any other
# model-specific transforms that may be needed by the model.
transform = Resize(size=(240, 240))
pre_processor = PreProcessor(transform=transform)
model = Padim(pre_processor=pre_processor)

# Better: inspect the default transforms before specifying custom
# transforms, and include the transforms that we don't want to modify.
print(Padim.configure_pre_processor().transform)
# Compose(
#       Resize(size=[256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True)
#       Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
# )

transform = Compose(
    Resize(size=(240, 240), interpolation=InterpolationMode.BILINEAR, antialias=True),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False),
)
pre_processor = PreProcessor(transform=transform)
model = Padim(pre_processor=pre_processor)

# Best: use the image_size argument in `configure_pre_processor` to directly
# obtain a PreProcessor instance with the right input size transform.
pre_processor = Padim.configure_pre_processor(image_size=(240, 240))
model = Padim(pre_processor=pre_processor)
```

```{seealso}
For more information about transforms:
- {doc}`Data Transforms Guide <../data/transforms>`
- {doc}`AnomalibModule Documentation <../../reference/models/base>`
```
