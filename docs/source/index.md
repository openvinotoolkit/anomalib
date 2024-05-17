# Anomalib Documentation

Anomalib is a deep learning library that aims to collect state-of-the-art anomaly detection algorithms for benchmarking on both public and private datasets. Anomalib provides several ready-to-use implementations of anomaly detection algorithms described in the recent literature, as well as a set of tools that facilitate the development and implementation of custom models. The library has a strong focus on visual anomaly detection, where the goal of the algorithm is to detect and/or localize anomalies within images or videos in a dataset. Anomalib is constantly updated with new algorithms and training/inference extensions, so keep checking!

:::{dropdown} {octicon}`checklist;1em`&nbsp; Key Features
:animate: fade-in-slide-down

- Simple and modular API and CLI for training, inference, benchmarking, and hyperparameter optimization.
- The largest public collection of ready-to-use deep learning anomaly detection algorithms and benchmark datasets.
- Lightning based model implementations to reduce boilerplate code and limit the implementation efforts to the bare essentials.
- The majority of models can be exported to OpenVINO Intermediate Representation (IR) for accelerated inference on Intel hardware.
- A set of inference tools for quick and easy deployment of the standard or custom anomaly detection models.
  :::

## {octicon}`package;1em` Installation

Anomalib provides two ways to install the library. The first is through PyPI, and the second is through a local installation. PyPI installation is recommended if you want to use the library without making any changes to the source code. If you want to make changes to the library, then a local installation is recommended.

::::{tab-set}

:::{tab-item} PyPI

```{literalinclude} ./snippets/install/pypi.txt
:language: bash
```

:::

:::{tab-item} Source

```{literalinclude} ./snippets/install/source.txt
:language: bash
```

:::

::::

## {octicon}`light-bulb` Get Started

::::{grid}

:::{grid-item-card} {octicon}`hourglass` Anomalib in 15 minutes
:link: markdown/get_started/anomalib
:link-type: doc

Get started with anomalib in 15 minutes.
:::

::::

## {octicon}`book` Guides

::::{grid}
:gutter: 1

:::{grid-item-card} {octicon}`codescan` Reference Guide
:link: markdown/guides/reference/index
:link-type: doc

Learn more about anomalib API and CLI.
:::

:::{grid-item-card} {octicon}`question` How-To Guide
:link: markdown/guides/how_to/index
:link-type: doc

Learn how to use anomalib for your anomaly detection tasks.
:::

:::{grid-item-card} {octicon}`telescope` Topic Guide
:link: markdown/guides/topic/index
:link-type: doc

Learn more about the internals of anomalib.
:::

:::{grid-item-card} {octicon}`code` Developer Guide
:link: markdown/guides/developer/index
:link-type: doc

Learn how to develop and contribute to anomalib.
:::

::::

```{toctree}
:caption: Get Started
:hidden:

markdown/get_started/anomalib
markdown/get_started/migration
```

```{toctree}
:caption: Guides
:hidden:

markdown/guides/reference/index
markdown/guides/how_to/index
markdown/guides/topic/index
markdown/guides/developer/index
```

```{toctree}
:caption: Announcements
:hidden:

markdown/announcements/recognition
markdown/announcements/engagement
```
