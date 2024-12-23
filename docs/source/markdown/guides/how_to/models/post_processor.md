```{eval-rst}
:orphan:
```

# Post-processing in Anomalib

This guide explains how post-processing works in Anomalib, its integration with models, and how to create custom post-processors.

## Overview

Post-processing in Anomalib is designed to handle model outputs and convert them into meaningful predictions. The post-processor:

- Computes anomaly scores from raw model outputs
- Determines optimal thresholds for anomaly detection
- Generates segmentation masks for pixel-level detection
- Gets exported with the model for consistent inference

The `PostProcessor` class is an abstract base class that serves two roles:

1. A PyTorch Module for processing model outputs that gets exported with the model
2. A Lightning Callback for managing thresholds during training

Anomalib provides concrete implementations like `OneClassPostProcessor` for specific use cases, such as one-class anomaly detection.
This is based on the `PostProcessor` class. For any other use case, you can create a custom post-processor by inheriting from the `PostProcessor` class.

## Basic Usage

The most common post-processor is `OneClassPostProcessor`:

```python
from anomalib.post_processing import OneClassPostProcessor

# Create a post-processor with sensitivity adjustments
post_processor = OneClassPostProcessor(
    image_sensitivity=0.5,    # Adjust image-level threshold sensitivity
    pixel_sensitivity=0.5     # Adjust pixel-level threshold sensitivity
)

# Apply to model outputs
predictions = post_processor(outputs)
print(predictions.pred_score)     # Normalized anomaly scores
print(predictions.pred_label)     # Binary predictions (0/1)
print(predictions.pred_mask)      # Segmentation masks (if applicable)
print(predictions.anomaly_map)    # Normalized anomaly maps
```

## Integration with Models

The post-processor is automatically integrated into Anomalib models:

```python
from anomalib.models import Patchcore
from anomalib.post_processing import OneClassPostProcessor

# Model creates default post-processor (OneClassPostProcessor)
model = Patchcore()

# Or specify custom post-processor
model = Patchcore(
    post_processor=OneClassPostProcessor(
        image_sensitivity=0.5,
        pixel_sensitivity=0.5
    )
)
```

## Creating Custom Post-processors

To create a custom post-processor, inherit from the abstract base class `PostProcessor`:

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

### Example: One-Class Post-processor

Here's a simplified version of how `OneClassPostProcessor` is implemented:

```python
from anomalib.post_processing import PostProcessor
from anomalib.data import InferenceBatch
from anomalib.metrics import F1AdaptiveThreshold, MinMax

class CustomOneClassPostProcessor(PostProcessor):
    """Custom one-class post-processor."""

    def __init__(
        self,
        image_sensitivity: float | None = None,
        pixel_sensitivity: float | None = None,
    ):
        super().__init__()
        self._image_threshold = F1AdaptiveThreshold()
        self._pixel_threshold = F1AdaptiveThreshold()
        self._image_normalization = MinMax()
        self._pixel_normalization = MinMax()

        self.image_sensitivity = image_sensitivity
        self.pixel_sensitivity = pixel_sensitivity

    def forward(self, predictions: InferenceBatch) -> InferenceBatch:
        """Post-process predictions."""
        if predictions.pred_score is None and predictions.anomaly_map is None:
            raise ValueError("At least one of pred_score or anomaly_map must be provided")

        # Normalize scores
        if predictions.pred_score is not None:
            predictions.pred_score = self._normalize(
                predictions.pred_score,
                self._image_normalization.min,
                self._image_normalization.max
            )
            predictions.pred_label = predictions.pred_score > self._image_threshold.value

        # Normalize anomaly maps
        if predictions.anomaly_map is not None:
            predictions.anomaly_map = self._normalize(
                predictions.anomaly_map,
                self._pixel_normalization.min,
                self._pixel_normalization.max
            )
            predictions.pred_mask = predictions.anomaly_map > self._pixel_threshold.value

        return predictions
```

## Best Practices

1. **Score Normalization**:

   - Normalize scores to [0,1] range
   - Handle numerical stability
   - Consider score distributions

2. **Threshold Selection**:

   - Use adaptive thresholding when possible
   - Validate thresholds on validation set
   - Consider application requirements

3. **Performance**:

   - Optimize computations for large outputs
   - Handle GPU/CPU transitions efficiently
   - Cache computed thresholds

4. **Validation**:
   - Verify prediction shapes
   - Check threshold computation
   - Test edge cases

## Common Pitfalls

1. **Threshold Issues**:

   - Not computing thresholds during training
   - Incorrect threshold computation
   - Not handling score distributions

2. **Normalization Problems**:

   - Inconsistent normalization
   - Numerical instability
   - Not handling outliers

3. **Memory Issues**:
   - Large intermediate tensors
   - Unnecessary CPU-GPU transfers
   - Memory leaks in custom implementations

## Edge Deployment

One key advantage of Anomalib's post-processor design is that it becomes part of the model graph during export. This means:

1. Post-processing is included in the exported OpenVINO model
2. No need for separate post-processing code in deployment
3. Consistent results between training and deployment

### Example: OpenVINO Deployment

```python
from anomalib.models import Patchcore
from anomalib.post_processing import OneClassPostProcessor
from openvino.runtime import Core
import numpy as np

# Training: Post-processor is part of the model
model = Patchcore(
    post_processor=OneClassPostProcessor(
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

### Benefits for Edge Deployment

1. **Simplified Deployment**:

   ```python
   # Before: Manual post-processing needed
   core = Core()
   model = core.read_model("model_without_postprocessing.xml")
   compiled_model = core.compile_model(model)
   raw_outputs = compiled_model([image])[output_key]
   normalized = normalize_scores(raw_outputs)
   predictions = apply_threshold(normalized)

   # After: Everything included in OpenVINO model
   core = Core()
   model = core.read_model("model.xml")
   compiled_model = core.compile_model(model)
   results = compiled_model([image])[output_key]  # Ready to use!
   ```

2. **Consistent Results**:

   - Same normalization across environments
   - Same thresholds as training
   - No implementation differences

3. **Optimized Performance**:

   - Post-processing operations are optimized by OpenVINO
   - Hardware acceleration for all operations
   - Reduced memory overhead
   - Fewer host-device transfers

4. **Reduced Deployment Complexity**:
   - No need to port post-processing code
   - Single model file contains everything
   - Simpler deployment pipeline
   - Ready for edge devices (CPU, GPU, VPU)

```{seealso}
For more information:
- {doc}`AnomalibModule Documentation <../../reference/models/base>`
- {doc}`Metrics Guide <../metrics/index>`
```
