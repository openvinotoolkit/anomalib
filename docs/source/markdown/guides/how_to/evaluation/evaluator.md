```{eval-rst}
:orphan:
```

# Evaluator

This guide explains how the Evaluator class works in Anomalib, its integration with metrics, and how to use it effectively.

## Prerequisites

- {doc}`Metrics <./metrics>`
- AnomalibModule
- Engine

## Overview

The Evaluator is a core component in Anomalib that:

- Computes and logs metrics during the validation and test sequence
- Integrates with PyTorch Lightning's training loop
- Manages metric computation across different devices (CPU/GPU)
- Provides flexibility in metric selection for validation and testing

The Evaluator serves as both:

1. A PyTorch Module for storing and organizing metrics
2. A Lightning Callback for metric computation during training

> **Note:**
> This guide assumes that you know how to create and use Anomalib metrics. If you are not familiar with this, please read the {doc}`Metrics How to Guide <./metrics>` first.

## Basic Usage

The Evaluator can be used to specify which metrics Anomalib should compute during your validation and/or training run. To achieve this, simply create some metrics (if you're unsure how to create metrics, please refer to the {doc}`Metrics How to Guide <./metrics>`), and pass them to a new `Evaluator` instance using either the `val_metrics` or the `test_metrics` argument, depending on in which stage of the pipeline you want the metrics to be used (of course, it's also possible to pass both validation and test metrics).

```python
from anomalib.metrics import F1Score, AUROC
from anomalib.metrics import Evaluator

# Initialize metrics with specific fields
f1_score = F1Score(fields=["pred_label", "gt_label"])
auroc = AUROC(fields=["pred_score", "gt_label"])

# Create evaluator with test metrics (for validation, use val_metrics arg)
evaluator = Evaluator(test_metrics=[f1_score, auroc])
```

To ensure that Anomalib uses your metrics during the testing sequence, the newly created evaluator instance should be passed to the model upon construction. For example, when we want to use the metrics to evaluate a Patchcore model:

```python
# Pass evaluator to model
model = Patchcore(
    evaluator=evaluator
)
```

That's it! Anomalib will now compute and report your metrics when running a testing sequence with your model. To trigger the testing sequence, simply call the `test` method of the engine and pass your model and the datamodule that contains your test set (if you are unsure how to create a datamodule, please refer to the {doc}`Datamodules How to Guide <../data/datamodules>`):

```python
from anomalib.engine import Engine

engine = Engine()
engine.test(model, datamodule=datamodule)  # make sure to create a datamodule first
```

## Stage-specific Metrics

You can configure different metrics for validation and testing:

```python
from anomalib.metrics import Evaluator, AUROC, F1Score

# Validation metrics
val_metrics = [
    AUROC(fields=["pred_score", "gt_label"]),     # Image-level AUROC
    F1Score(fields=["pred_label", "gt_label"])    # Image-level F1
]

# Test metrics (more comprehensive)
test_metrics = [
    AUROC(fields=["pred_score", "gt_label"]),     # Image-level AUROC
    AUROC(fields=["anomaly_map", "gt_mask"]),     # Pixel-level AUROC
    F1Score(fields=["pred_label", "gt_label"]),   # Image-level F1
    F1Score(fields=["pred_mask", "gt_mask"])      # Pixel-level F1
]

# Create evaluator with both sets
evaluator = Evaluator(
    val_metrics=val_metrics,
    test_metrics=test_metrics
)

# Use with model
model = Patchcore(evaluator=evaluator)
```

## Device Management

The Evaluator manages metric computation across devices. By default, the metrics are computed on CPU to save GPU memory. To enforce metric computation on the same device as your model and data, you can set the `compute_on_cpu` argument to `False`. This will also ensure that the internal states of all metric instances will be stored on the same device as the model.

```python
# Compute on CPU (default)
evaluator = Evaluator(
    test_metrics=metrics,
    compute_on_cpu=True  # Default
)

# Compute on same device as model
evaluator = Evaluator(
    test_metrics=metrics,
    compute_on_cpu=False
)
```

> **Note:**
> For multi-GPU training, `compute_on_cpu` is automatically set to `False`.

## Best Practices

### 1. Strategic Metric Selection

Choose metrics based on your specific use case and requirements:

```python
# Image Classification Task
image_metrics = [
    AUROC(fields=["pred_score", "gt_label"]),     # Overall detection performance
    F1Score(fields=["pred_label", "gt_label"]),   # Balance between precision and recall
]

# Segmentation Task
segmentation_metrics = [
    AUROC(fields=["pred_score", "gt_label"]),     # Image-level detection
    AUROC(fields=["anomaly_map", "gt_mask"]),     # Pixel-level detection accuracy
    F1Score(fields=["pred_mask", "gt_mask"]),     # Segmentation quality
    PRO(fields=["anomaly_map", "gt_mask"])        # Region-based evaluation
]

# Multi-task Evaluation
evaluator = Evaluator(
    test_metrics=[
        *image_metrics,          # Image-level metrics
        *segmentation_metrics    # Pixel-level metrics
    ]
)
```

### 2. Efficient Resource Management

Balance between accuracy and computational efficiency:

```python
# Memory-Efficient Configuration
evaluator = Evaluator(
    # Validation: Light-weight metrics for quick feedback
    val_metrics=[
        F1Score(fields=["pred_label", "gt_label"]),
        AUROC(fields=["pred_score", "gt_label"])
    ],
    # Testing: Comprehensive evaluation
    test_metrics=[
        F1Score(fields=["pred_label", "gt_label"]),
        AUROC(fields=["pred_score", "gt_label"]),
        PRO(fields=["anomaly_map", "gt_mask"]),    # Compute-intensive metric
    ],
    # Move computation to CPU for large datasets
    compute_on_cpu=True
)
```

```{seealso}
For more information:
- {doc}`Metrics Documentation <../../reference/metrics/index>`
- {doc}`AnomalibModule Guide <../models/anomalib_module>`
```
