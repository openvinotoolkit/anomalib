```{eval-rst}
:orphan:
```

# Data Transforms

This guide will explain how Anomalib applies transforms to the input images, and how these transforms can be configured for various use-cases.

## Prerequisites

- [Torchvision Transforms](https://pytorch.org/vision/stable/transforms.html)
- {doc}`Datasets <./datasets>`
- {doc}`Datamodules <./datamodules>`

## Overview

Data transforms are operations that are applied to the raw input images before they are passed to the model. In Anomalib, we distinguish between two types of transforms:

- **Model-specific transforms** that convert the input images to the format expected by the model.
- **Data augmentations** for dataset enrichment and increasing the effective sample size.

After reading this guide, you will understand the difference between these two transforms, and know when and how to use both types of transform.

```{note}
Anomalib uses the [Torchvision Transforms v2 API](https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html) to apply transforms to the input images. Before reading this guide, please make sure that you are familiar with the basic principles of Torchvision transforms.
```

## Model-specific transforms

Most vision models make some explicit assumptions about the format of the input images. For example, the model may be configured to read the images in a specific shape, or the model may expect the images to be normalized to the mean and standard deviation of the dataset on which the backbone was pre-trained. These type of transforms are tightly coupled to the chosen model architecture, and need to be applied to any image that is passed to the model. In Anomalib, we refer to these transforms as "model-specific transforms".

### Default model-specific transforms

Model-specific transforms in Anomalib are defined in the model implementation, and applied by the {doc}`PreProcessor <../models/pre_processor>`. To ensure that the right transforms are applied to the input images, each Anomalib model is required to implement the `configure_pre_processor` class, which returns a default `PreProcessor` instance that contains the model-specific transforms. These transforms will be applied to any input images before passing the images to the model, unless a custom set of model-specific transforms is passed by the user (see {ref}`custom_model_transforms`).

We can inspect the default pre-processor of the `Padim` model to find the default set of model-specific transforms for this model:

```python
from anomalib.models import Padim

pre_processor = Padim.configure_pre_processor()
print(pre_processor.transform)

# Compose(
#       Resize(size=[256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True)
#       Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
# )
```

As we can see, Padim's default set of transforms consists of a `Resize` transform to resize the images to an input shape of 256x256 pixels, followed by a `Normalize` transform to normalize the images to the mean and standard deviation of the ImageNet dataset.

(custom_model_transforms)=

### Custom model-specific transforms

In some cases it may be desired to change the model-specific transforms. For example, we may want to increase the input resolution of the images or change the normalization statistics to reflect a different pre-training dataset. To achieve this, we can define a new set of transforms, wrap the transforms in a new `PreProcessor` instance, and pass the pre-processor when instantiating the model:

```python
from anomalib.models import Padim
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Normalize, Resize

transform = Compose([
    Resize(size=(512, 512)),
    Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),  # CLIP stats
])

pre_processor = PreProcessor(transform=transform)
model = Padim(pre_processor=pre_processor)
```

The most common use-case for custom model-specific transforms is varying the input size. Most Anomalib models are largely invariant to the shape of the input images, so we can freely change the size of the Resize transform. To accommodate this use-case, the Lightning model's `configure_pre_processor` method allows passing an optional `image_size` argument, which updates the size of the `Resize` transform from its default value. This allows us to easily obtain a pre-processor instance which transforms the images to the new input shape, but in which the other model-specific transforms are unmodified.

```python
from anomalib.models import Padim

pre_processor = Padim.configure_pre_processor(image_size=(240, 360))
model = Padim(pre_processor=pre_processor)

print(model.pre_processor.transform)
# Compose(
#       Resize(size=[240, 360], interpolation=InterpolationMode.BILINEAR, antialias=True)
#       Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
# )
```

For models that require a fixed input size, such as WinClip, passing an image size to the `configure_pre_processor` method won't work. These models will notify the user that the input size of the model cannot be changed, and use the default, required input size instead.

```python
from anomalib.models import WinClip

pre_processor = WinClip.configure_pre_processor(image_size=(240, 360))
# WARNING:anomalib.models.image.winclip.lightning_model:Image size is not used in WinCLIP. The input image size is determined by the model.

print(pre_processor.transform)
# Compose(
#       Resize(size=[240, 240], interpolation=InterpolationMode.BICUBIC, antialias=True)
#       Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711], inplace=False)
# )
```

```{note}
Some caution is required when passing custom model-specific transforms. Models may have some strict requirements for their input images which could be violated when using a custom set of transforms. Always make sure that you understand the model's input requirements before changing the model-specific transforms!
```

### Export and inference

For consistent model behaviour in inference settings, it is important that the appropriate model-specific transforms are applied in the model deployment stage. To facilitate this, Anomalib infuses the model-specific transforms in the model graph when exporting models to ONNX and OpenVINO. This saves the user the effort of transforming the input images in their inference pipeline, and mitigates the risk of inconsistent input transforms between training/validation and inference.

As shown in the following example, defining a custom transform and passing it to the model is sufficient to ingrain the transforms in the exported model graph (of course, the same principle applies when using the default model-specific transforms).

```python
from torchvision.transforms.v2 import Resize
from anomalib.pre_processing import PreProcessor


transform = Resize((112, 112))
pre_processor = PreProcessor(transform=transform)
model = MyModel(pre_processor=pre_processor)

model.to_onnx("model.onnx")  # the resize transform is included in the ONNX model
```

The `Resize` transform will get added to the exported model graph, and applied to the input images during inference. This greatly simplifies the deployment workflow, as no explicit pre-processing steps are needed anymore. You can just pass the raw images directly to the model, and the model will transform the input images before passing them to the first layer of the model.

## Data augmentations

Data augmentation refers to the practice of applying transforms to input images to increase the variability in the dataset. By transforming the images, we effectively increase the sample size which helps improve a model's generalization and robustness to variations in real-world scenarios. Augmentations are often randomized to maximize variability between training runs and/or epochs. Some common augmentations include flipping, rotating, or scaling images, adjusting brightness or contrast, adding noise, and cropping.

In Anomalib, data augmentations are configured from the `DataModule` and applied by the `Dataset`. Augmentations can be configured separately for each of the subsets (train, val, test) to suit different use-cases such as training set enrichment or test-time augmentations (TTA). All datamodules in Anomalib have the `train_augmentations`, `val_augmentations` and `test_augmentations` arguments, to which the user can pass a set of augmentation transforms. The following example shows how to add some random augmentations to the training set of an MVTecAD dataset:

```python
from anomalib.data import MVTecAD
from torchvision.transforms import v2

augmentations = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),   # Randomly flip images horizontally with 50% probability
    v2.RandomVerticalFlip(p=0.2),     # Randomly flip images vertically with 20% probability
    v2.RandomRotation(degrees=30),    # Randomly rotate images within a range of ±30 degrees
    v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Randomly crop and resize images
    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # Randomly adjust colors
    v2.RandomGrayscale(p=0.1),        # Convert images to grayscale with 10% probability
])

datamodule = MVTecAD(
    category="transistor",
    train_augmentations=augmentations,
    val_augmentations=None,
    test_augmentations=None,
    augmentations=None,  # use this argument to set train, val and test augmentations simultaneously
)
```

In this example, the datamodule will pass the provided training augmentations to the dataset instance that holds the training samples. The transforms will be applied to each image when the dataloader fetches the images from the dataset.

Note that unlike model-specific transforms, data augmentations will not be included in the model graph during export. Please take this into consideration when deciding which type of transform is most suitable for your use-case when designing your data pipeline.

## Additional resizing in collate

In some rare cases, an additional resize operation may be applied to the input images when the `Dataloader` collates the input images into a batch. This only happens when the dataset contains images of different shapes, and neither the data augmentations nor the model-specific transforms contain a `Resize` transform. In this case, the collate method resizes all images to a common shape as a safeguard to prevent shape mismatch error when concatenating. The images will be resized to the dimensions of the image within the batch with the largest width or height.

Note that this is not desirable, as the user has no control over the interpolation method and antialiasing setting of the resize operation. For this reason, it is advised to always include a resize transform in the model-specific transforms.

## Common pitfalls

### 1. Passing a model-specific transforms as augmentation

Anomalib expects un-normalized images from the dataset, so that any Normalize transforms present in the model-specific transforms get applied correctly. Adding a Normalize transform to the augmentations will lead to unexpected behaviour as the model-specific transform will apply an additional normalization operation.

```python
# Wrong: by passing the Normalize transform as an augmentation, the transform will not
# be included in the model graph during export. The model may also apply its default Normalize
# transform, leading to incorrect and unpredictable behaviour.
augmentations = Compose(
    RandomHorizontalFlip(p=0.5),
    Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
)
datamodule = MVTecAD(train_augmentations=augmentations)
model = Padim()
engine = Engine()
engine.fit(model, datamodule=datamodule)

# Correct: pass the random flip as an augmentation to the datamodule, and pass the updated
# Normalize transform to a new PreProcessor instance.
augmentations = RandomHorizontalFlip(p=0.5)
datamodule = MVTecAD(train_augmentations=augmentations)

transform = Compose(
    Resize(size=(256, 256)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
)
pre_processor = PreProcessor(transform=transform)
model = Padim(pre_processor=pre_processor)

engine = Engine()
engine.fit(model, datamodule=datamodule)
```

Similarly, adding a Resize transform to the augmentations with the intention of changing the input size of the images will not have the desired effect. Any Resize transform present in the model-specific transforms will overrule the Resize from the augmentations.

```python
# Wrong: The resize in the augmentations will be overruled by the resize in the
# model-specific transforms. The final image size will not be 224x224, but 256x256,
# as dictated by the default model-specific transforms.
augmentations = Compose(
    RandomHorizontalFlip(p=0.5),
    Resize(size=(224, 224)),  # overruled by resize in default model-specific transform
)
datamodule = MVTecAD(augmentations=augmentations)

model = Padim()

engine = Engine()
engine.fit(model, datamodule=datamodule)

# Correct: pass the random flip as an augmentation to the datamodule, and pass an
# updated pre-processor instance with the new image shape to the model. The final
# image size will be 224x224.
augmentations = RandomHorizontalFlip(p=0.5)
datamodule = MVTecAD(augmentations=augmentations)

pre_processor = Padim.configure_pre_processor(image_size=(224, 224))
model = Padim(pre_processor=pre_processor)

engine = Engine()
engine.fit(model, datamodule=datamodule)
```

### 2. Passing an augmentation as model-specific transform

Passing an augmentation transform to the `PreProcessor` can have unwanted effects. The augmentation will be included in the model graph, so during inference we will apply random horizontal flips. Since the PreProcessor defines a single transform for all stages, the random flips will also be applied during validation and testing.

```python
# Wrong: Augmentation transform added to model-specific transforms
transform = Compose(
    RandomHorizontalFlip(p=0.5),
    Resize(size=(256, 256)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
)
pre_processor = PreProcessor(transform=transform)
model = Padim(pre_processor=pre_processor)

datamodule = MVTecAD()

engine = Engine()
engine.fit(model, datamodule=datamodule)

# Correct: Pass the transform to the datamodule as `train_augmentation`.
augmentations = RandomHorizontalFlip(p=0.5)
datamodule = MVTecAD(train_augmentation=augmentations)

model = Padim()

engine = Engine()
engine.fit(model, datamodule=datamodule)
```

```{seealso}
For more information:
- {doc}`PreProcessor Guide <../models/pre_processor>`
- {doc}`DataModules Guide <./datamodules>`
- {doc}`Datasets Guide<./datasets>`
```
