# Custom Data

This tutorial will show you how to train anomalib models on your custom
data. More specifically, we will show you how to use the [Folder](../../reference/data/image/folder.md)
dataset to train anomalib models on your custom data.

```{warning}
This tutorial assumes that you have already installed anomalib.
If not, please refer to the installation section.
```

```{note}
We will use our [hazelnut_toy](https://github.com/openvinotoolkit/anomalib/releases/download/hazelnut_toy_dataset/hazelnut_toy.zip) dataset to show the capabilities of the [Folder](../../reference/data/image/folder.md)
dataset, but you can use any dataset you want.
```

We will split the section to two tasks: Classification and Segmentation.

## Classification Dataset

In certain use-cases, ground-truth masks for the abnormal images may not be
available. In such cases, we could use the classification task to train a model
that will be able to detect the abnormal images in the test set.

We will split this section into two tasks:

- Classification with normal and abnormal images, and
- Classification with only normal images.

### With Normal and Abnormal Images

We could use [Folder](../../reference/data/image/folder.md) datamodule to train
a model on this dataset. We could run the following python code to create the
custom datamodule:

:::::{dropdown} Code Syntax
:icon: code

::::{tab-set}
:::{tab-item} API
:sync: label-1

```{literalinclude} ../../../../snippets/data/image/folder/classification/default.txt
:language: python
```

```{note}
As can be seen above, we only need to specify the ``task`` argument to ``classification``. We could have also use ``TaskType.CLASSIFICATION`` instead of ``classification``.
```

The [Folder](../../reference/data/image/folder.md) datamodule will create training, validation, test and
prediction datasets and dataloaders for us. We can access the datasets
and dataloaders by following the same approach as in the segmentation
task.

When we check the samples from the dataloaders, we will see that the
`mask` key is not present in the samples. This is because we do not need
the masks for the classification task.

```{literalinclude} ../../../../snippets/data/image/folder/classification/dataloader_values.txt
:language: python
```

Training the model is as simple as running the following command:

```{literalinclude} ../../../../snippets/train/api/classification/model_and_engine.txt
:language: python
```

where we train a Patchcore model on this custom dataset with default model parameters.

:::

:::{tab-item} CLI
:sync: label-2

Here is the dataset config to create the same custom datamodule:

```{literalinclude} ../../../../snippets/config/data/image/folder/classification/cli/default.yaml
:language: yaml
```

Assume that we have saved the above config file as `classification.yaml`. We could run the following CLI command to train a Patchcore model on above dataset:

```bash
anomalib train --data classification.yaml --model anomalib.models.Patchcore --task CLASSIFICATION
```

```{note}
As can be seen above, we also need to specify the ``task`` argument to ``CLASSIFICATION`` to explicitly tell anomalib that we want to train a classification model. This is because the default `task` is `SEGMENTATION` within `Engine`.
```

:::
::::
:::::

### With Only Normal Images

There are certain cases where we only have normal images in our dataset but
would like to train a classification model.

This could be done in two ways:

- Train the model and skip the validation and test steps, as we do not have
  abnormal images to validate and test the model on, or
- Use the synthetic anomaly generation feature to create abnormal images from
  normal images, and perform the validation and test steps.

For now we will focus on the second approach.

#### With Validation and Testing via Synthetic Anomalies

If we want to check the performance of the model, we will need to have abnormal
images to validate and test the model on. During the validation stage,
these anomalous images are used to normalize the anomaly scores and find the
best threshold that separates normal and abnormal images.

Anomalib provides synthetic anomaly generation capabilities to create abnormal
images from normal images so we could check the performance. We could use the
[Folder](../../reference/data/image/folder.md) datamodule to train a model on
this dataset.

:::::{dropdown} Code Syntax
:icon: code

::::{tab-set}
:::{tab-item} API
:sync: label-1

```{literalinclude} ../../../../snippets/data/image/folder/classification/normal_and_synthetic.txt
:language: python
```

Once the datamodule is setup, the rest of the process is the same as in the previous classification example.

```{literalinclude} ../../../../snippets/train/api/classification/model_and_engine.txt
:language: python
```

where we train a Patchcore model on this custom dataset with default model parameters.

:::

:::{tab-item} CLI
:sync: label-2

Here is the CLI command to create the same custom datamodule with only normal images. We only need to change the `test_split_mode` argument to `SYNTHETIC` to generate synthetic anomalies.

```{literalinclude} ../../../../snippets/config/data/image/folder/classification/cli/normal_and_synthetic.yaml
:language: yaml
```

Assume that we have saved the above config file as `normal.yaml`. We could run the following CLI command to train a Patchcore model on above dataset:

```bash
anomalib train --data normal.yaml --model anomalib.models.Patchcore --task CLASSIFICATION
```

```{note}
As shown in the previous classification example, we, again, need to specify the ``task`` argument to ``CLASSIFICATION`` to explicitly tell anomalib that we want to train a classification model. This is because the default `task` is `SEGMENTATION` within `Engine`.
```

:::
::::
:::::

## Segmentation Dataset

Assume that we have a dataset in which the training set contains only
normal images, and the test set contains both normal and abnormal
images. We also have masks for the abnormal images in the test set. We
want to train an anomaly segmentation model that will be able to detect the
abnormal regions in the test set.

### With Normal and Abnormal Images

We could use [Folder](../../reference/data/image/folder.md) datamodule to load the hazelnut dataset in a format that is readable by Anomalib's models.

:::::{dropdown} Code Syntax
::::{tab-set}
:::{tab-item} API
:sync: label-1

We could run the following python code to create the custom datamodule:

```{literalinclude} ../../../../snippets/data/image/folder/segmentation/default.txt
:language: python
```

The [Folder](../../reference/data/image/folder.md) datamodule will create training, validation, test and
prediction datasets and dataloaders for us. We can access the datasets
and dataloaders using the following attributes:

```{literalinclude} ../../../../snippets/data/image/folder/segmentation/datamodule_attributes.txt
:language: python
```

To check what individual samples from dataloaders look like, we can run
the following command:

```{literalinclude} ../../../../snippets/data/image/folder/segmentation/dataloader_values.txt
:language: python
```

We could check the shape of the images and masks using the following
commands:

```python
print(train_data["image"].shape)
# torch.Size([32, 3, 256, 256])

print(train_data["mask"].shape)
# torch.Size([32, 256, 256])
```

Training the model is as simple as running the following command:

```{literalinclude} ../../../../snippets/train/api/segmentation/model_and_engine.txt
:language: python
```

where we train a Patchcore model on this custom dataset with default model parameters.
:::

:::{tab-item} CLI
:sync: label-2
Here is the CLI command to create the same custom datamodule:

```{literalinclude} ../../../../snippets/config/data/image/folder/segmentation/cli/default.yaml
:language: yaml
```

Assume that we have saved the above config file as `segmentation.yaml`.
We could run the following CLI command to train a Patchcore model on above dataset:

```bash
anomalib train --data segmentation.yaml --model anomalib.models.Patchcore
```

:::

::::
:::::

This example demonstrates how to create a segmentation dataset with
normal and abnormal images. We could expand this example to create a
segmentation dataset with only normal images.

### With Only Normal Images

There are certain cases where we only have normal images in our dataset
but would like to train a segmentation model. This could be done in two ways:

- Train the model and skip the validation and test steps, as
  we do not have abnormal images to validate and test the model on, or
- Use the synthetic anomaly generation feature to create abnormal
  images from normal images, and perform the validation and test steps.

For now we will focus on the second approach.

#### With Validation and Testing via Synthetic Anomalies

We could use the synthetic anomaly generation feature again to create abnormal
images from normal images. We could then use the
[Folder](../../reference/data/image/folder.md) datamodule to train a model on
this dataset. Here is the python code to create the custom datamodule:

:::::{dropdown} Code Syntax
:icon: code

::::{tab-set}
:::{tab-item} API
:sync: label-1

We could run the following python code to create the custom datamodule:

```{literalinclude} ../../../../snippets/data/image/folder/segmentation/normal_and_synthetic.txt
:language: python
```

As can be seen from the code above, we only need to specify the
`test_split_mode` argument to `SYNTHETIC`. The [Folder](../../reference/data/image/folder.md) datamodule will create training, validation, test and prediction datasets and
dataloaders for us.

To check what individual samples from dataloaders look like, we can run
the following command:

```{literalinclude} ../../../../snippets/data/image/folder/segmentation/dataloader_values.txt
:language: python
```

We could check the shape of the images and masks using the following
commands:

```python
print(train_data["image"].shape)
# torch.Size([32, 3, 256, 256])

print(train_data["mask"].shape)
# torch.Size([32, 256, 256])
```

:::

:::{tab-item} CLI
:sync: label-2

Here is the CLI command to create the same custom datamodule with only normal
images. We only need to change the `test_split_mode` argument to `SYNTHETIC` to
generate synthetic anomalies.

```{literalinclude} ../../../../snippets/config/data/image/folder/segmentation/cli/normal_and_synthetic.yaml
:language: yaml
```

Assume that we have saved the above config file as `synthetic.yaml`. We could
run the following CLI command to train a Patchcore model on above dataset:

```bash
anomalib train --data synthetic.yaml --model anomalib.models.Patchcore
```

:::
::::
:::::
