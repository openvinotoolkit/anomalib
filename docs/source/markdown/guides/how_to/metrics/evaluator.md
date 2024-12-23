```{eval-rst}
:orphan:
```

# Evaluator in Anomalib

This guide explains how the Evaluator class works in Anomalib, its integration with metrics, and how to use it effectively.

## Overview

The Evaluator is a core component in Anomalib that:

- Computes and logs metrics during validation and test steps
- Integrates with PyTorch Lightning's training loop
- Manages metric computation across different devices (CPU/GPU)
- Provides flexibility in metric selection for validation and testing

The Evaluator serves as both:

1. A PyTorch Module for organizing metrics
2. A Lightning Callback for metric computation during training

## Field-Based Metrics

Anomalib's metrics are designed to be field-based, which means they operate on specific fields from your data batch. This approach offers several advantages:

1. **Flexible Configuration**:

   ```python
   # Compute AUROC between prediction scores and ground truth
   auroc_score = AUROC(fields=["pred_score", "gt_label"])

   # Same metric can be used for different fields
   auroc_pixel = AUROC(fields=["anomaly_map", "gt_mask"])
   ```

2. **Multiple Evaluation Targets**:

   ```python
   # Evaluate both image and pixel-level metrics
   metrics = [
       F1Score(fields=["pred_label", "gt_label"]),      # Image-level
       F1Score(fields=["pred_mask", "gt_mask"])         # Pixel-level
   ]
   ```

3. **Custom Field Mapping**:

   ```python
   # Your model might use different field names
   metrics = AUROC(fields={
       "preds": "anomaly_scores",    # Map 'preds' to your 'anomaly_scores'
       "target": "ground_truth"      # Map 'target' to your 'ground_truth'
   })
   ```

This field-based approach differs from standard TorchMetrics in several ways:

1. **Configuration vs Hard-coding**:

   ```python
   # TorchMetrics: Fixed input structure
   metric = torchmetrics.AUROC()
   metric(preds, target)  # Must follow this format

   # Anomalib: Flexible field mapping
   metric = AUROC(fields=["custom_pred", "custom_target"])
   metric.update(batch)  # Can use any batch structure
   ```

2. **Batch-level Access**:

   ```python
   # TorchMetrics: Limited to specific inputs
   metric(predictions, targets)

   # Anomalib: Access to entire batch
   metric.update(batch)  # Can access any field in the batch
   ```

3. **Multi-modal Support**:

   ```python
   # Support for different data modalities
   image_metric = AUROC(fields=["image_pred", "image_gt"])
   video_metric = AUROC(fields=["video_pred", "video_gt"])
   depth_metric = AUROC(fields=["depth_pred", "depth_gt"])
   ```

## Basic Usage

Here's a simple example of using the Evaluator:

```python
from anomalib.metrics import F1Score, AUROC
from anomalib.metrics import Evaluator
from anomalib.data import ImageBatch
import torch

# Initialize metrics with specific fields
f1_score = F1Score(fields=["pred_label", "gt_label"])
auroc = AUROC(fields=["pred_score", "gt_label"])

# Create evaluator with test metrics
evaluator = Evaluator(test_metrics=[f1_score, auroc])

# Create sample batch
batch = ImageBatch(
    image=torch.rand(4, 3, 256, 256),
    pred_label=torch.tensor([0, 0, 1, 1]),
    gt_label=torch.tensor([0, 0, 1, 1]),
    pred_score=torch.tensor([0.1, 0.2, 0.8, 0.9])
)

# Metrics are automatically updated during training
# You don't need to call these manually
evaluator.on_test_batch_end(None, None, None, batch, 0)
evaluator.on_test_epoch_end(None, None)
```

## Integration with Models

The Evaluator needs to be explicitly created and passed to Anomalib models:

```python
from anomalib.models import Patchcore
from anomalib.metrics import AUROC, F1Score, Evaluator

# Create metrics
metrics = [
    AUROC(fields=["pred_score", "gt_label"]),
    F1Score(fields=["pred_label", "gt_label"])
]

# Create evaluator with metrics
evaluator = Evaluator(test_metrics=metrics)

# Pass evaluator to model
model = Patchcore(
    evaluator=evaluator
)
```

## Configuring Metrics

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

The Evaluator intelligently manages metric computation across devices:

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

Note: For multi-GPU training, `compute_on_cpu` is automatically set to `False`.

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

### 3. Consistent Evaluation Strategy

Maintain consistency across experiments for fair comparison:

```python
# Define standard metric set
def get_standard_metrics(task_type="classification"):
    """Get standardized metrics for consistent evaluation."""
    base_metrics = [
        AUROC(fields=["pred_score", "gt_label"], name="image_auroc"),
        F1Score(fields=["pred_label", "gt_label"], name="image_f1")
    ]

    if task_type == "segmentation":
        base_metrics.extend([
            AUROC(fields=["anomaly_map", "gt_mask"], name="pixel_auroc"),
            F1Score(fields=["pred_mask", "gt_mask"], name="pixel_f1")
        ])

    return base_metrics

# Use consistent metrics across experiments
model1 = Patchcore(
    evaluator=Evaluator(test_metrics=get_standard_metrics("segmentation"))
)
model2 = EfficientAd(
    evaluator=Evaluator(test_metrics=get_standard_metrics("segmentation"))
)
```

### 4. Proper Metric State Management

Handle metric states correctly in different scenarios:

```python
from lightning.pytorch import Trainer

# Reset metrics between runs
def train_multiple_runs(model, num_runs=3):
    trainer = Trainer()
    for run in range(num_runs):
        # Reset evaluator for each run
        model.evaluator.val_metrics.reset()
        model.evaluator.test_metrics.reset()
        trainer.fit(model)
        trainer.test(model)

# Separate metrics for different datasets
def evaluate_multiple_datasets(model, datasets):
    results = {}
    for dataset_name, dataset in datasets.items():
        # Create new evaluator for each dataset
        model.evaluator = Evaluator(
            test_metrics=get_standard_metrics(),
            compute_on_cpu=True
        )
        results[dataset_name] = trainer.test(model, dataset)
    return results
```

## Common Pitfalls

### 1. Incorrect Field Specifications

Field mismatches might be a common source of errors:

```python
# Wrong: Mismatched field names
metrics = [
    AUROC(fields=["predictions", "labels"]),           # Wrong names
    F1Score(fields=["anomaly_scores", "gt_labels"])    # Wrong names
]

# Wrong: Missing required fields
metrics = [
    AUROC(fields=["pred_score"]),      # Missing ground truth field
    F1Score(fields=["pred_label"])     # Missing ground truth field
]

# Correct: Match your data batch fields
batch = ImageBatch(
    image=torch.rand(32, 3, 224, 224),
    pred_score=torch.rand(32),
    pred_label=torch.randint(2, (32,)),
    gt_label=torch.randint(2, (32,))
)

metrics = [
    AUROC(fields=["pred_score", "gt_label"]),     # Matches batch fields
    F1Score(fields=["pred_label", "gt_label"])    # Matches batch fields
]
```

### 2. Resource Management Issues

Common memory and computation problems you might encounter:

```python
# Wrong: GPU Memory Overflow
evaluator = Evaluator(
    test_metrics=[
        PRO(fields=["anomaly_map", "gt_mask"]),    # Memory-intensive
        AUROC(fields=["anomaly_map", "gt_mask"]),  # Large pixel maps
    ],
    compute_on_cpu=False  # Forces GPU computation
)

# Better: Memory-Efficient Configuration
evaluator = Evaluator(
    test_metrics=[
        PRO(fields=["anomaly_map", "gt_mask"]),
        AUROC(fields=["anomaly_map", "gt_mask"]),
    ],
    compute_on_cpu=True  # Moves computation to CPU
)

# Best: Staged Evaluation
def evaluate_in_stages(model, dataloader):
    # Stage 1: Quick metrics
    model.evaluator = Evaluator(
        test_metrics=[
            AUROC(fields=["pred_score", "gt_label"]),
            F1Score(fields=["pred_label", "gt_label"])
        ],
        compute_on_cpu=False  # Fast GPU computation for light metrics
    )
    quick_results = trainer.test(model, dataloader)

    # Stage 2: Memory-intensive metrics
    model.evaluator = Evaluator(
        test_metrics=[
            PRO(fields=["anomaly_map", "gt_mask"])
        ],
        compute_on_cpu=True  # CPU computation for heavy metrics
    )
    detailed_results = trainer.test(model, dataloader)

    return {**quick_results, **detailed_results}
```

### 3. Distributed Training Pitfalls

Handle distributed scenarios correctly:

```python
# Wrong: Manual metric synchronization
class IncorrectDistributedMetric(AnomalibMetric):
    def compute(self):
        # Don't manually gather/reduce metrics
        gathered = torch.distributed.gather(self.value)
        return gathered.mean()

# Correct: Let Lightning handle synchronization
class CorrectDistributedMetric(AnomalibMetric):
    def __init__(self, fields):
        super().__init__(
            fields,
            dist_sync_on_step=False,  # Sync only on compute()
            compute_on_cpu=True       # Avoid GPU OOM in distributed setting
        )

    def compute(self):
        # Lightning handles the synchronization
        return self.value

# Correct: Distributed training setup
trainer = pl.Trainer(
    devices=4,
    strategy="ddp",
    sync_batchnorm=True  # Important for consistent normalization
)

evaluator = Evaluator(
    test_metrics=[CorrectDistributedMetric(fields=["pred_score", "gt_label"])],
    compute_on_cpu=True  # Recommended for distributed training
)
```

### 4. Validation/Test Metric Confusion

Avoid mixing validation and test metrics:

```python
# Wrong: Same heavy metrics for validation and testing
evaluator = Evaluator(
    val_metrics=[
        PRO(fields=["anomaly_map", "gt_mask"]),    # Too heavy for validation
        AUROC(fields=["anomaly_map", "gt_mask"]),
        F1Score(fields=["pred_mask", "gt_mask"])
    ],
    test_metrics=[
        PRO(fields=["anomaly_map", "gt_mask"]),
        AUROC(fields=["anomaly_map", "gt_mask"]),
        F1Score(fields=["pred_mask", "gt_mask"])
    ]
)

# Better: Appropriate metrics for each phase
evaluator = Evaluator(
    # Validation: Quick feedback metrics
    val_metrics=[
        AUROC(fields=["pred_score", "gt_label"]),     # Fast image-level metric
        F1Score(fields=["pred_label", "gt_label"])    # Quick binary evaluation
    ],
    # Testing: Comprehensive evaluation
    test_metrics=[
        AUROC(fields=["pred_score", "gt_label"]),     # Image-level AUROC
        AUROC(fields=["anomaly_map", "gt_mask"]),     # Pixel-level AUROC
        F1Score(fields=["pred_label", "gt_label"]),   # Image-level F1
        F1Score(fields=["pred_mask", "gt_mask"]),     # Pixel-level F1
        PRO(fields=["anomaly_map", "gt_mask"])        # Detailed region analysis
    ]
)
```

```{seealso}
For more information:
- {doc}`Metrics Documentation <../../reference/metrics/index>`
- {doc}`AnomalibModule Guide <../models/anomalib_module>`
```
