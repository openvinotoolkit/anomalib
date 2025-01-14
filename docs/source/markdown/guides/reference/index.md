# Reference Guide

This section contains the API and CLI reference for anomalib.

## Core Components

::::{grid} 2 2 2 3
:gutter: 2
:padding: 1

:::{grid-item-card} {octicon}`database` Data
:link: ./data/index
:link-type: doc

Core component for data handling and datasets.
:::

:::{grid-item-card} {octicon}`dependabot` Models
:link: ./models/index
:link-type: doc

Anomaly detection model implementations.
:::

:::{grid-item-card} {octicon}`gear` Engine
:link: ./engine/index
:link-type: doc

Core training and inference engine.
:::
::::

## Processing & Analysis

::::{grid} 2 2 2 3
:gutter: 2
:padding: 1

:::{grid-item-card} {octicon}`filter` Pre-processing
:link: ./pre_processing/index
:link-type: doc

Data preparation and augmentation.
:::

:::{grid-item-card} {octicon}`filter` Post-processing
:link: ./post_processing/index
:link-type: doc

Anomaly map processing and thresholding.
:::

:::{grid-item-card} {octicon}`meter` Metrics
:link: ./metrics/index
:link-type: doc

Performance evaluation metrics.
:::
::::

## Framework Components

::::{grid} 2 2 2 3
:gutter: 2
:padding: 1

:::{grid-item-card} {octicon}`graph` Loggers
:link: ./loggers/index
:link-type: doc

Experiment logging and tracking.
:::

:::{grid-item-card} {octicon}`gear` Callbacks
:link: ./callbacks/index
:link-type: doc

Training callbacks and hooks.
:::

:::{grid-item-card} {octicon}`workflow` Pipelines
:link: ./pipelines/index
:link-type: doc

Training and optimization pipelines.
:::

:::{grid-item-card} {octicon}`image` Visualization
:link: ./visualization/index
:link-type: doc

Result visualization tools.
:::

:::{grid-item-card} {octicon}`tools` Utils
:link: ./utils/index
:link-type: doc

Utility functions and helpers.
:::

:::{grid-item-card} {octicon}`terminal` CLI
:link: ./cli/index
:link-type: doc

Command line interface tools.
:::
::::

::::{grid} 1
:gutter: 2
:padding: 1

:::{grid-item-card} {octicon}`cpu` Inference
:link: ./deploy/index
:link-type: doc

Model inference and optimization.
:::
::::

```{toctree}
:caption: Reference
:hidden:

./data/index
./models/index
./engine/index
./pre_processing/index
./post_processing/index
./metrics/index
./loggers/index
./callbacks/index
./pipelines/index
./visualization/index
./utils/index
./cli/index
./deploy/index
```
