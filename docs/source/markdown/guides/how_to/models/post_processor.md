```{eval-rst}
:orphan:
```

# Post-processing in Anomalib

This guide explains how post-processing works in Anomalib, its integration with models, and how to create custom post-processors.

## Overview

Post-processing in Anomalib refers to any additional operations that are applied after the model generates its raw predictions. Most anomaly detection models do not generate hard classification labels directly. Instead, the models generate an anomaly score, which can be seen as an estimation of the distance from the sample to the learned representation of normality. The raw anomaly scores may consist of a single score per image for anomaly classification, or a pixel-level anomaly map for anomaly localization/segmentation. The raw anomaly scores may be hard to interpret, as they are unbounded, and the range of values may differ between models. To convert the raw anomaly scores into useable predictions, we need to apply a threshold that maps the raw scores to the binary (normal vs. anomalous) classification labels. In addition, we may want to normalize the raw scores to the [0, 1] range for interpretability and visualization.

The thresholding and normalization steps described above are typical post-processing steps in an anomaly detection workflow. The module that is responsible for these operations in Anomalib is the `PostProcessor`. The `PostProcessor` applies a set of post-processing operations on the raw predictions returned by the model. Similar to the {doc}`PreProcessor <./pre_processor>`, the `PostProcessor` also infuses its operations in the model graph during export. This ensures that during deployment:

- Post-processing is part of the exported model (ONNX, OpenVINO)
- Users don't need to manually apply post-processing steps such as thresholding and normalization
- Edge deployment is simplified with automatic post-processing

To achieve this, the `PostProcessor` class implements the following components:

1. A PyTorch Module for processing model outputs that gets exported with the model
2. A Lightning Callback for managing thresholds during training

## PostProcessor

The `PostProcessor` is Anomalib's default post-processor class which covers the most common anomaly detection workflow. It is responsible for adaptively computing the optimal threshold value for the dataset, applying this threshold during testing/inference, and normalizing the predicted anomaly scores to the [0, 1] range for interpretability. Thresholding and normalization is applied separately for both image- and pixel-level predictions. The following descriptions focus on the image-level predictions, but the same principles apply for the pixel-level predictions.

**Thresholding**

The post-processor adaptively computes the optimal threshold value during the validation sequence. The threshold is computed by collecting the raw anomaly scores and the corresponding ground truth labels for all the images in the validation set, and plotting the Precision-Recall (PR) curve for the range of possible threshold values $\mathbf{\theta}$.

The resulting precision and recall values are then used to calculate the F1-score for each threshold value ${\theta}_i$ using the following formula:

$$
F1_i = 2 \times \frac{Precision(\theta_i) × Recall(\theta_i)}{Precision(\theta_i) + Recall(\theta_i)}
$$

Finally, the optimal threshold value $\theta^*$ is determined as the threshold value that yields the highest the F1-score:

$$
\theta^* = \text{arg}\max_{i} F1_{i}
$$

During testing and predicting, the post-processor computes the binary classification labels by assigning a positive label (anomalous) to all anomaly scores that are higher than the threshold, and a negative label (normal) to all anomaly scores below the threshold. Given an anomaly score $s_{\text{test},i}$, the binary classifical label $\hat{y}_{\text{test},i}$ is given by:

$$
\hat{y}_{\text{test},i} =
\begin{cases}
1 & \text{if } s_{\text{test},i} \geq \theta^* \\
0 & \text{if } s_{\text{test},i} < \theta^*
\end{cases}
$$

**Normalization**

During the validation sequence, the post-processor iterates over the raw anomaly score predictions for the validation set, $\mathbf{s}_{\text{val}}$, and keeps track of the lowest and highest observed values, $\min\mathbf{s}_{\text{val}}$ and $\max \mathbf{s}_{\text{val}}$.

During testing and predicting, the post-processor uses the stored min and max values, together with the optimal threshold value, to normalize the values to the [0, 1] range. For a raw anomaly score $s_{\text{test},i}$, the normalized score $\tilde{s}_{\text{test},i}$ is given by:

$$
\tilde{s}_{\text{test},i} = \frac{s_{\text{test},i} - \theta^*}{\max\mathbf{s}_\text{val} - \min\mathbf{s}_\text{val}} + 0.5
$$

As a last step, the normalized scores are capped between 0 and 1.

The $\theta^*$ term in the formula above ensures that the normalized values are centered around the threshold value, such that a value of 0.5 in the normalized domain corresponds to the value of the threshold in the un-normalized domain. This helps with interpretability of the results, as it asserts that normalized values of 0.5 and higher are labeled anomalous, while values below 0.5 are labeled normal.

Centering the threshold value around 0.5 has the additional advantage that it allows us to add a sensitivity parameter $\alpha$ that changes the sensitivity of the anomaly detector. In the normalized domain, the binary classification label is given by:

$$
\hat{y}_{\text{test},i} =
\begin{cases}
1 & \text{if } \tilde{s}_{\text{test},i} \geq 1 - \alpha \\
0 & \text{if } \tilde{s}_{\text{test},i} < 1 - \alpha
\end{cases}
$$

Where $\alpha$ is a sensitivity parameter that can be varied between 0 and 1, such that a higher sensitivity value lowers the effective anomaly score threshold. The sensitivity parameter can be tuned depending on the use case. For example, use-cases in which false positives should be avoided may benefit from reducing the sensitivity.

```{note}
Normalization and thresholding only works when your datamodule contains a validation set, preferably cosisting of both normal and anomalous samples. When your validation set only contains normal samples, the threshold will be set to the value of the highest observed anomaly score in your validation set.
```

## Basic Usage

To use the `PostProcessor`, simply add it to any Anomalib model when creating the model:

```python
from anomalib.models import Padim
from anomalib.post_processing import PostProcessor

post_processor = PostProcessor()
model = Padim(post_processor=post_processor)
```

The post-processor can be configured using its constructor arguments. In the case of the `PostProcessor`, the only configuration parameters are the sensitivity for the thresholding operation on the image- and pixel-level:

```python
post_processor = PostProcessor(
    image_sensitivity=0.4,
    pixel_sensitivity=0.4,
)
model = Padim(post_processor=post_processor)
```

When a post-processor instance is not passed explicitly to the model, the model will automatically configure a default post-processor instance. Let's confirm this by creating a Padim model and printing the `post_processor` attribute:

```python
model = Padim()
print(model.post_processor)
# PostProcessor(
#   (_image_threshold): F1AdaptiveThreshold() (value=0.50)
#   (_pixel_threshold): F1AdaptiveThreshold() (value=0.50)
#   (_image_normalization_stats): MinMax()
#   (_pixel_normalization_stats): MinMax()
# )
```

Each model implementation in Anomalib is required to implement the `configure_post_processor` method, which defines the default post-processor for that model. We can use this method to quickly inspect the default post-processing behaviour of an Anomalib model class:

```python
print(Padim.configure_post_processor())
```

In some cases it may be desirable to disable post-processing entirely. This is done by passing `False` to the model's `post_processor` argument:

```python
from anomalib.models import Padim

model = Padim(post_processor=False)
print(model.post_processor is None)  # True
```

### Exporting

One key advantage of Anomalib's post-processor design is that it becomes part of the model graph during export. This means:

1. Post-processing is included in the exported OpenVINO model
2. No need for separate post-processing code in deployment
3. Consistent results between training and deployment

### Example: OpenVINO Deployment

```python
from anomalib.models import Patchcore
from anomalib.post_processing import PostProcessor
from openvino.runtime import Core
import numpy as np

# Training: Post-processor is part of the model
model = Patchcore(
    post_processor=PostProcessor(
        image_sensitivity=0.5,
        pixel_sensitivity=0.5
    )
)

# Export: Post-processing is included in the graph
model.export("model", export_mode="openvino")

# Deployment: Simple inference without manual post-processing
core = Core()
ov_model = core.read_model("model.xml")
compiled_model = core.compile_model(ov_model)

# Get input and output names
input_key = compiled_model.input(0)
output_key = compiled_model.output(0)

# Prepare input
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Run inference - everything is handled by the model
results = compiled_model([image])[output_key]

# Results are ready to use
anomaly_maps = results[..., 0]     # Already normalized maps
pred_scores = results[..., 1]      # Already normalized scores
pred_labels = results[..., 2]      # Already thresholded (0/1)
pred_masks = results[..., 3]       # Already thresholded masks (if applicable)
```

## Creating Custom Post-processors

Advanced users may want to define their own post-processing pipeline. This can be useful when the default post-processing behaviour of the `PostProcessor` is not suitable for the model and its predictions. To create a custom post-processor, inherit from `nn.Module` and `Callback`, and implement your post-processing logic using lightning hooks. Don't forget to also include the post-processing steps in the `forward` method of your class to ensure that the post-processing is included when exporting your model:

```python
from anomalib.post_processing import PostProcessor
from anomalib.data import InferenceBatch
import torch

class CustomPostProcessor(PostProcessor):
    """Custom post-processor implementation."""

    def forward(self, predictions: InferenceBatch) -> InferenceBatch:
        """Post-process predictions.

        This method must be implemented by all subclasses.
        """
        # Implement your post-processing logic here
        raise NotImplementedError
```

After defining the class, it can be used in any Anomalib workflow by passing it to the model:

```python
from anomalib.models import Padim

post_processor = CustomPostProcessor()
model = Padim(post_processor=post_processor)
```

## Best Practices

**Validation**:

- Ensure that your validation set contains both normal and anomalous samples.
- Ensure that your validation set contains sufficient representative samples.

```{seealso}
For more information:
- {doc}`PreProcessing guide <./pre_processing>`
- {doc}`AnomalibModule Documentation <../../reference/models/base>`
```
