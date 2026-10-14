# Metrics

Anomalib provides a comprehensive set of metrics for evaluating anomaly detection model performance. All metrics extend TorchMetrics' functionality with Anomalib-specific features.

## Available Metrics

### Area Under Curve Metrics

::::{grid} 2
:gutter: 2

:::{grid-item-card} AUROC
:link: auroc
:link-type: doc

Area Under the Receiver Operating Characteristic curve. Measures the model's ability to distinguish between normal and anomalous samples.
:::

:::{grid-item-card} AUPR
:link: aupr
:link-type: doc

Area Under the Precision-Recall curve. Particularly useful for imbalanced datasets.
:::

:::{grid-item-card} AUPRO
:link: aupro
:link-type: doc

Area Under the Per-Region Overlap curve. Evaluates pixel-level anomaly localization performance.
:::

:::{grid-item-card} AUPIMO
:link: aupimo
:link-type: doc

Area Under the Per-Image Missed Overlap curve. Advanced metric for evaluating localization quality.
:::

::::

### F1 Score Metrics

::::{grid} 2
:gutter: 2

:::{grid-item-card} F1Score
:link: f1_score
:link-type: doc

Standard F1 score for binary classification. Harmonic mean of precision and recall.
:::

:::{grid-item-card} F1Max
:link: f1_score
:link-type: doc

Maximum F1 score across all possible thresholds. Useful for finding optimal operating points.
:::

::::

### Threshold Metrics

::::{grid} 2
:gutter: 2

:::{grid-item-card} F1AdaptiveThreshold
:link: threshold
:link-type: doc

Automatically determines the optimal threshold by maximizing F1 score.
:::

:::{grid-item-card} ManualThreshold
:link: threshold
:link-type: doc

Uses a manually specified threshold for classification.
:::

::::

### Other Metrics

::::{grid} 2
:gutter: 2

:::{grid-item-card} PRO
:link: pro
:link-type: doc

Per-Region Overlap score for evaluating pixel-level localization.
:::

:::{grid-item-card} PIMO
:link: pimo
:link-type: doc

Per-Image Missed Overlap for assessing localization errors.
:::

:::{grid-item-card} PGn
:link: pg_pb
:link-type: doc

Presorted Good with n% bad samples missed. Measures false negative rate at specific operating points.
:::

:::{grid-item-card} PBn
:link: pg_pb
:link-type: doc

Presorted Bad with n% good samples misclassified. Measures false positive rate at specific operating points.
:::

:::{grid-item-card} MinMax
:link: min_max
:link-type: doc

Normalizes anomaly scores to [0, 1] range using min-max scaling.
:::

:::{grid-item-card} AnomalyScoreDistribution
:link: anomaly_score_distribution
:link-type: doc

Analyzes and tracks the distribution of anomaly scores for model diagnostics.
:::

::::

### Utility Classes

::::{grid} 2
:gutter: 2

:::{grid-item-card} AnomalibMetric
:link: base
:link-type: doc

Base class for all Anomalib metrics. Extends TorchMetrics with field-based updates.
:::

:::{grid-item-card} Evaluator
:link: evaluator
:link-type: doc

Orchestrates multiple metrics for comprehensive model evaluation.
:::

:::{grid-item-card} BinaryPrecisionRecallCurve
:link: precision_recall_curve
:link-type: doc

Computes precision-recall curves for binary classification tasks.
:::

::::

## API Reference

```{toctree}
:caption: Metric Reference
:hidden:

auroc
aupr
aupro
aupimo
f1_score
threshold
pro
pimo
pg_pb
min_max
anomaly_score_distribution
base
evaluator
precision_recall_curve
```

```{eval-rst}
.. automodule:: anomalib.metrics
   :members: AUROC, AUPR, AUPRO, AUPIMO, F1Score, F1Max, F1AdaptiveThreshold, ManualThreshold, PRO, PIMO, PGn, PBn, MinMax, AnomalyScoreDistribution, AnomalibMetric, Evaluator, BinaryPrecisionRecallCurve, create_anomalib_metric
   :undoc-members:
   :show-inheritance:
   :no-index:
```
