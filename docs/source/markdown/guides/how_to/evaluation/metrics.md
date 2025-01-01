```{eval-rst}
:orphan:
```

# Metrics

This guide explains how to use and configure Anomalib's Evaluation metrics to rate the performance of Anomalib models.

## Preprequisites

- {doc}`Dataclasses <../data/dataclasses>`
- Torchmetrics

## Overview

Metric computation in Anomalib is built around the `AnomalibMetric` class, which acts as an extension of TorchMetrics' `Metric` class. `AnomalibMetric` adds Anomalib-specific functionalities to integrate seamlessly with Anomalib's dataclasses and improve ease-of-use within various parts the library.

## Field-Based Metrics

The main difference between standard `TorchMetrics` classes and `AnomalibMetric` classes is the addition of the `fields` argument in the latter. When instantiating an `AnomalibMetric` subclass, the user has to specify which fields from Anomalib's dataclasses should be used when updating the metric. When `update` is called, the user can pass a dataclass instance directly, and the metric will automatically fetch the required fields from the instance.

Consider the following example which computes the image-level Area Under the ROC curve (AUROC) given a set of batch predictions. The example shows both the classical `TorchMetrics` approach, and the new `AnomalibMetric` approach to illustrate the difference between the two.

```python
# standard torch metric
from torchmetrics import AUROC
auroc = AUROC()
for batch in predictions:
    auroc.update(batch.pred_label, gt_label)
print(auroc.compute())  # tensor(0.94)

# anomalib version of metric
from anomalib.metrics import AUROC
auroc = AUROC(fields=["pred_label", "gt_label"])
for batch in predictions:
    auroc.update(batch)
print(auroc.compute())  # tensor(0.94)
```

This may look like a trivial difference, but directly passing the batch to the update method greatly simplifies evaluation pipelines, as we don't need to keep track of which type of predictions need to be passed to which metric. Instead, the metric itself holds this information and fetches the appropriate fields from the batch when its update method is called.

For example, we can use Anomalib's metric class to compute both image- and pixel-level AUROC. Note how we don't need to pass the image- and pixel-level predictions explicitly when iterating over the batches.

```python
from anomalib.metrics import AUROC

# prefix is optional, but useful to distinguish between two metrics of the same type
image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")

# name that will be used by Lightning when logging the metrics
print(image_auroc.name)  # 'image_AUROC'
print(pixel_auroc.name)  # 'pixel_AUROC'

for batch in predictions:
    image_auroc.update(batch)
    pixel_auroc.update(batch)
print(image_auroc.compute())  # tensor(0.98)
print(pixel_auroc.compute())  # tensor(0.96)
```

### Creating a new AnomalibMetric class

Anomalib's `metrics` module provides Anomalib versions of various performance metrics commonly used in anomaly detection, such as `AUROC`, `AUPRO` and `F1Score`. In addition, any subclass of `Metric` can easily be converted into an `AnomalibMetric`, as shown below:

```python
from torchmetrics import Accuracy  # metric that we want to convert

# option 1: Define the new class explicitly
class AnomalibAccuracy(AnomalibMetric, Accuracy):
    pass

# option 2: use the helper function
AnomalibAccuracy = create_anomalib_metric(Accuracy)

# after creating the new class, we gain access to AnomalibMetric's extended functinality
accuracy = AnomalibAccuracy(fields=["pred_label", "gt_label"])
accuracy.update(batch)
print(accuracy.compute())  # tensor(0.76)
```

Note that we still have access to all the constructor arguments of the original metric. For example, we can configure the Accuracy metric created above to compute either the micro average or the macro average:

```python
from torchmetrics import Accuracy
from anomalib.metrics import create_anomalib_metric

# create the Anomalib metric
AnomalibAccuracy = create_anomalib_metric(Accuracy)

# instantiate with different init args
micro_acc = AnomalibAccuracy(fields=["pred_label", "gt_label"], average="micro")
macro_acc = AnomalibAccuracy(fields=["pred_label", "gt_label"], average="macro")

# update and compute the metrics
for batch in predictions:
    micro_acc.update(batch)
    macro_acc.update(batch)
print(micro_acc.compute())  # tensor(0.87)
print(macro_acc.compute())  # tensor(0.79)
```

## Usage in Anomalib pipeline

Anomalib provides an {doc}`Evaluator <./evaluator>` class to facilitate metric computation. The evaluator takes care of all the aspects of metric computation, including updating and computing the metrics, and logging the final metric values.

To include a set of metrics to an Anomalib pipeline, simply wrap them in an evaluator instance, and pass it to the model using the `evaluator` argument, for example:

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

When `Engine.test()` is called, the `Evaluator` will ensure that all metrics get updated and that the final metric values are computed and logged at the end of the testing sequence.

Note that specifying custom evaluation metrics is optional. By default, each model defines a default set of metrics that will be computed when nothing is specified by the user.

For a more detailed description and more examples of the `Evaluator` class, please visit the {doc}`Evaluator How to Guide <./evaluator>`.

## Common Pitfalls

### 1. No use of prefixes when using metrics of same type

Adding a prefix to your metric name helps avoid problems with Lightning's metric logging:

```python
from anomalib.metrics import F1Score

# Wrong: Same type metrics without prefix will have same name
image_f1 = F1Score(fields=["pred_label", "gt_label"])
pixel_f1 = F1Score(fields=["pred_mask", "gt_mask"])
print(image_f1.name)  # F1Score
print(pixel_f1.name)  # F1Score

# Correct: Prefixes will ensure unique metric names
image_f1 = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
pixel_f1 = F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_")
print(image_f1.name)  # 'image_F1Score'
print(pixel_f1.name)  # 'pixel_F1Score'
```

### 2. Incorrect Field Specifications

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

```{seealso}
For more information:
- {doc}`Evaluator Documentation <./evaluator>`
- {doc}`AnomalibModule Guide <../models/anomalib_module>`
```
