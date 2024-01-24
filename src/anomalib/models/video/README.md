# Anomalib Video Models

## 📝 Description

This sub-package contains the models for handling video datasets in anomalib.

The anomalib.models.video subpackage provides:

- Classes and functions to define video anomaly models.
- Models for video-based anomaly classification, detection or segmentation.

## ⚠️ Note

The models defined here are designed specifically to handle video datasets
These models contain spatio-temporal layers that are not present in the image
models.

## 💡 Examples

The following example shows how to use the AiVad model to train on the Avenue dataset.

<details>
<summary>Training the AiVad model on Avenue video dataset</summary>

```python
# Import the necessary modules
from anomalib.data import Avenue
from anomalib.models import AiVad
from anomalib.engine import Engine

# Load the avenue dataset, model and engine.
datamodule = Avenue()
model = AiVad()
engine = Engine()

# Train the model
engine.train(model, datamodule)
```

</details>
