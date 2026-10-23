# Anomalib Image Models

## 📝 Description

This sub-package contains the models for handling image datasets in anomalib.

The anomalib.models.image subpackage provides:

- Classes and functions to define image anomaly models.
- Models for image-based anomaly classification, detection or segmentation.

## ⚠️ Note

The models in anomalib.models.image can also handle video datasets by converting them to frame-based image datasets.
This feature allows the application of the same models and techniques to video data.

## 💡 Examples

<details>
<summary>Using the EfficientAd model on an Image Dataset such as Visa</summary>

```python
# Import the necessary modules
from anomalib.data import Visa
from anomalib.models import EfficientAd
from anomalib.engine import Engine

# Load the ViSA dataset, model and engine.
# EfficientAd requires a training batch size of 1.
datamodule = Visa(train_batch_size=1)
model = EfficientAd()
engine = Engine()

# Train the model
engine.train(model, datamodule)
```

</details>

<details>
<summary>Using the EfficientAd model on a Video Dataset such as Avenue</summary>

To use an image model to train on a video dataset, we need to convert the video dataset to a frame-based image dataset. To do this, we could use `clip_length_in_frames=1` when loading the dataset.

```python
# Import the necessary modules
from anomalib.data import Avenue
from anomalib.models import EfficientAd
from anomalib.engine import Engine

# Load the folder, model and engine.
# Set the clip_length_in_frames to 1 to convert the video dataset to a
#   frame-based image dataset.
# EfficientAd requires a training batch size of 1.
datamodule = Avenue(clip_length_in_frames=1, train_batch_size=1)
model = EfficientAd()
engine = Engine()

# Train the model
engine.train(model, datamodule)
```

</details>
