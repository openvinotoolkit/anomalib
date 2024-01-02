# Custom Data

This tutorial will show you how to train anomalib models on your custom data. More specifically, we will show you how to use the `Folder` dataset to train anomalib models on your custom data.

```{warning}
This tutorial assumes that you have already installed anomalib.
If not, please refer to the :ref:`installation` section.
```

```{note}
We will use the MVTec AD dataset to show the capabilities of the ``Folder`` dataset, but you can use any dataset you want.
```

We will split the section to two task: Classification and Segmentation.

## Segmentation Task

Assume that we have a dataset in which the training set contains only normal images, and the test set contains both normal and abnormal images. We also have masks for the abnormal images in the test set. We want to train a segmentation model that will be able to detect the abnormal regions in the test set.

### How to Create a Segmentation Dataset with Normal and Abnormal Images

We could use `Folder` datamodule to train a model on this dataset.

::::{tab-set}
:::{tab-item} API
:sync: label-1

We could run the following python code to create the custom datamodule:

```python
from anomalib.data import Folder

# Create the datamodule
datamodule = Folder(
    root="datasets/MVTec/transistor",
    normal_dir="train/good",
    abnormal_dir="test/bent_lead",
    mask_dir="ground_truth/bent_lead",
)

# Setup the datamodule
datamodule.setup()
```

The `Folder` datamodule will create training, validation, test and prediction datasets and dataloaders for us. We can access the datasets and dataloaders using the following attributes:

```python
# Access the datasets
train_dataset = datamodule.train_data
val_dataset = datamodule.val_data
test_dataset = datamodule.test_data

# Access the dataloaders
train_dataloader = datamodule.train_dataloader()
val_dataloader = datamodule.val_dataloader()
test_dataloader = datamodule.test_dataloader()
```

To check what individual samples from dataloaders look like, we can run the following command:

```python
i, train_data = next(enumerate(datamodule.train_dataloader()))
print(train_data.keys())
# dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

i, val_data = next(enumerate(datamodule.val_dataloader()))
print(val_data.keys())
# dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

i, test_data = next(enumerate(datamodule.test_dataloader()))
print(test_data.keys())
# dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])
```

We could check the shape of the images and masks using the following commands:

```python
print(train_data["image"].shape)
# torch.Size([32, 3, 256, 256])

print(train_data["mask"].shape)
# torch.Size([32, 256, 256])
```

:::

:::{tab-item} CLI
:sync: label-2

Add the cli code here.

:::

::::

This example demonstrates how to create a segmentation dataset with normal and abnormal images. We could expand this example to create a segmentation dataset with only normal images.

### How to Create a Segmentation Dataset with only Normal Images?

There are certain cases where we only have normal images in our dataset but would like to train a segmentation model. Anomalib provides synthetic anomaly generation capabilities to create abnormal images from normal images. We could use the `Folder` datamodule to train a model on this dataset.

:::::{dropdown} Syntax
:icon: code

::::{tab-set}
:::{tab-item} API
:sync: label-1

We could run the following python code to create the custom datamodule:

```python
from anomalib.data.utils import TestSplitMode


datamodule = Folder(
    root="datasets/MVTec/transistor",
    normal_dir="train/good",
    test_split_mode=TestSplitMode.SYNTHETIC,
)
```

As can be seen from the code above, we only need to specify the `test_split_mode` argument to `SYNTHETIC`. The `Folder` datamodule will create training, validation, test and prediction datasets and dataloaders for us. We can access the datasets and dataloaders using the following attributes:

```python
# Access the datasets
train_dataset = datamodule.train_data
val_dataset = datamodule.val_data
test_dataset = datamodule.test_data

# Access the dataloaders
train_dataloader = datamodule.train_dataloader()
val_dataloader = datamodule.val_dataloader()
test_dataloader = datamodule.test_dataloader()
```

To check what individual samples from dataloaders look like, we can run the following command:

```python
i, train_data = next(enumerate(datamodule.train_dataloader()))
print(train_data.keys())
# dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

i, val_data = next(enumerate(datamodule.val_dataloader()))
print(val_data.keys())
# dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

i, test_data = next(enumerate(datamodule.test_dataloader()))
print(test_data.keys())
# dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])
```

We could check the shape of the images and masks using the following commands:

```python
print(train_data["image"].shape)
# torch.Size([32, 3, 256, 256])

print(train_data["mask"].shape)
# torch.Size([32, 256, 256])
```

:::

:::{tab-item} CLI
:sync: label-2

Add the cli code here.

:::
::::
:::::

This example demonstrates how to create a segmentation dataset with only normal images.

## Classification Task

Assume that we have a dataset in which the training set contains only normal images, and the test set contains both normal and abnormal images. We want to train a classification model that will be able to detect the abnormal images in the test set.

### How to Create a Classification Dataset with Normal and Abnormal Images

We could use `Folder` datamodule to train a model on this dataset. We could run the following python code to create the custom datamodule:

```python
from anomalib.data import Folder


# Create the datamodule for the classification task
datamodule = Folder(
    root="datasets/MVTec/transistor",
    normal_dir="train/good",
    abnormal_dir="test/bent_lead",
    task="classification",
)
```

```{note}
As can be seen above, we only need to specify the ``task`` argument to ``classification``. We could have also use ``TaskType.CLASSIFICATION`` instead of ``classification``.
```

The `Folder` datamodule will create training, validation, test and prediction datasets and dataloaders for us. We can access the datasets and dataloaders by following the same approach as in the segmentation task.

When we check the samples from the dataloaders, we will see that the `mask` key is not present in the samples. This is because we do not need the masks for the classification task.

```python
i, train_data = next(enumerate(datamodule.train_dataloader()))
print(train_data.keys())
# dict_keys(['image_path', 'label', 'image'])

i, val_data = next(enumerate(datamodule.val_dataloader()))
print(val_data.keys())
# dict_keys(['image_path', 'label', 'image'])

i, test_data = next(enumerate(datamodule.test_dataloader()))
print(test_data.keys())
# dict_keys(['image_path', 'label', 'image'])
```

### How to Create a Classification Dataset with only Normal Images?

Similar to the segmentation task, there are certain cases where we only have normal images in our dataset but would like to train a classification model. We could use the synthetic anomaly generation feature again to create abnormal images from normal images. We could then use the `Folder` datamodule to train a model on this dataset. Here is the python code to create the custom datamodule:

```python
from anomalib.data.utils import TestSplitMode


datamodule = Folder(
    root="datasets/MVTec/transistor",
    normal_dir="train/good",
    test_split_mode=TestSplitMode.SYNTHETIC,
    task="classification",
)
```
