# Anomalib Image Models

## 📝 Description

This sub-package contains the models for handling image datasets in anomalib.

The anomalib.models.image subpackage provides:
Classes and functions for working with image datasets.
Models for image classification, object detection, and image generation.

## ⚠️ Note

The models in anomalib.models.image can also handle video datasets by converting them to frame-based image datasets.
This feature allows the application of the same models and techniques to video data.

## 💡 Examples

<details>
<summary>Using the EfficientAD model on an Image Dataset such as Visa</summary>

```python
# Import the necessary modules
from anomalib.data import Visa
from anomalib.models import EfficientAD
from anomalib.engine import Engine

# Load the ViSA dataset, model and engine.
datamodule = Visa()
model = EfficientAD()
engine = Engine()

# Train the model
engine.train(model, datamodule)
```

</details>

<details>
<summary>Using the EfficientAD model on a Video Dataset such as Visa</summary>

To use an image model to train on a video dataset, we need to convert the video dataset to a frame-based image dataset. To do this, we could use Anomalib's Folder dataset.

```python
# Import the necessary modules
from anomalib.data import Folder
from anomalib.models import EfficientAD
from anomalib.engine import Engine

# Load the folder, model and engine.
datamodule = Folder(
    normal_dir="path/to/normal/frames",
    abnormal_dir="path/to/abnormal/frames",
    mask_dir="path/to/mask/frames",
    image_size=(224, 224),
)
model = EfficientAD()
engine = Engine()

# Train the model
engine.train(model, datamodule)
```

</details>
