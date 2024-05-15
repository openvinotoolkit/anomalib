# Data Transforms

This tutorial will show how Anomalib applies transforms to the input images, and how these transforms can be configured. Anomalib uses the [Torchvision Transforms v2 API](https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html) to apply transforms to the input images.

Common transforms are the `Resize` transform, which is used to resize the input images to a fixed width and height, and the `Normalize` transform, which normalizes the pixel values of the input images to a pre-determined range. The normalization statistics are usually chosen to correspond to the pre-training characteristics of the model's backbone. For example, when the backbone of the model was pre-trained on ImageNet dataset, it is usually recommended to normalize the model's input images to the mean and standard deviation of the pixel values of ImageNet. In addition, there are many other transforms which could be useful to achieve the desired pre-processing of the input images and to apply data augmentations during training.

## Using custom transforms for training and evaluation

When we create a new datamodule, it will not be equipped with any transforms by default. When we load an image from the datamodule, it will have the same shape and pixel values as the original image from the file system.

```{literalinclude} ../../../../snippets/data/transforms/datamodule_default.txt
:language: python
```

Now let's create another datamodule, this time passing a simple resize transform to the datamodule using the `transform` argument.

::::{tab-set}
:::{tab-item} API
:sync: label-1

```{literalinclude} ../../../../snippets/data/transforms/datamodule_custom.txt
:language: python
```

:::

:::{tab-item} CLI
:sync: label-2

In the CLI, we can specify a custom transforms by providing the class path and init args of the Torchvision transforms class:

```{literalinclude} ../../../../snippets/data/transforms/datamodule_custom_cli.yaml
:language: yaml
```

::::

As we can see, the datamodule now applies the custom transform when loading the images, resizing both training and test data to the specified shape.

In the above example, we used the `transform` argument to assign a single set of transforms to be used both in the training and in the evaluation subsets. In some cases, we might want to apply distinct sets of transforms between training and evaluation. This can be useful, for example, when we want to apply random data augmentations during training to improve generalization of our model. Using different transforms for training and evaluation can be done easily by specifying different values for the `train_transform` and `eval_transform` arguments. The train transforms will be applied to the images in the training subset, while the eval transforms will be applied to images in the validation, testing and prediction subsets.

::::{tab-set}
:::{tab-item} API
:sync: label-1

```{literalinclude} ../../../../snippets/data/transforms/datamodule_train_eval.txt
:language: python
```

:::

:::{tab-item} CLI
:sync: label-2

`train_transform` and `eval_transform` can also be set separately from CLI. Note that the CLI also supports stacking multiple transforms using a `Compose` object.

```{literalinclude} ../../../../snippets/data/transforms/datamodule_train_eval_cli.yaml
:language: yaml
```

::::

```{note}
Please note that it is not recommended to pass only one of `train_transform` and `eval_transform` while keeping the other parameter empty. This could lead to unexpected behaviour, as it might lead to a mismatch between the training and testing subsets in terms of image shape and normalization characteristics.
```

## Model-specific transforms

Each Anomalib model defines a default set of transforms, that will be applied to the input data when the user does not specify any custom transforms. The default transforms of a model can be inspected using the `configure_transforms` method, for example:

```{literalinclude} ../../../../snippets/data/transforms/model_configure.txt
:language: python
```

As shown in the example, the default transforms for PatchCore consist of resizing the image to 256x256 pixels, followed by center cropping to an image size of 224x224. Finally, the pixel values are normalized to the mean and standard deviation of the ImageNet dataset. These transforms correspond to the recommended pre-processing steps described in the original PatchCore paper.

The use of these model-specific transforms ensures that Anomalib automatically applies the right transforms when no custom transforms are passed to the datamodule by the user. When no user-defined transforms are passed to the datamodule, Anomalib's engine assigns the model's default transform to the `train_transform` and `eval_transform` of the datamodule at the start of the fit/val/test sequence:

::::{tab-set}
:::{tab-item} API
:sync: label-1

```{literalinclude} ../../../../snippets/data/transforms/model_fit.txt
:language: python
```

:::

:::{tab-item} CLI
:sync: label-2

Since the CLI uses the Anomalib engine under the hood, the same principles concerning model-specific transforms apply when running a model from the CI. Hence, the following command will ensure that Patchcore's model-specific default transform is used when fitting the model.

```{literalinclude} ../../../../snippets/data/transforms/model_fit_cli.sh
:language: bash
```

::::

## Transforms during inference

To ensure consistent transforms between training and inference, Anomalib includes the eval transform in the exported model. During inference, the transforms are infused in the model's forward pass which ensures that the transforms are always applied. The following example illustrates how Anomalib's torch inferencer automatically applies the transforms stored in the model. The same principles apply to both Lightning inference and OpenVINO inference.

::::{tab-set}
:::{tab-item} API
:sync: label-1

```{literalinclude} ../../../../snippets/data/transforms/inference.txt
:language: python
```

:::

:::{tab-item} CLI
:sync: label-2

The CLI behaviour is equivalent to that of the API. When a model is trained with a custom `eval_transform` like in the example below, the `eval_transform` is included both in the saved lightning model as in the exported torch model.

```{literalinclude} ../../../../snippets/data/transforms/inference_cli.yaml
:language: yaml
```

```{literalinclude} ../../../../snippets/data/transforms/inference_cli.sh
:language: bash
```

::::

:::
::::
:::::
