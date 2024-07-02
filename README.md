<div align="center">

<img src="https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/_static/images/logos/anomalib-wide-blue.png" width="600px" alt="Anomalib Logo">

**A library for benchmarking, developing and deploying deep learning anomaly detection algorithms**

---

[Key Features](#key-features) •
[Docs](https://anomalib.readthedocs.io/en/latest/) •
[Notebooks](notebooks) •
[License](LICENSE)

[![python](https://img.shields.io/badge/python-3.7%2B-green)]()
[![pytorch](https://img.shields.io/badge/pytorch-1.8.1%2B-orange)]()
[![openvino](https://img.shields.io/badge/openvino-2022.3.0-purple)]()

[![Pre-Merge Checks](https://github.com/openvinotoolkit/anomalib/actions/workflows/pre_merge.yml/badge.svg)](https://github.com/openvinotoolkit/anomalib/actions/workflows/pre_merge.yml)
[![Documentation Status](https://readthedocs.org/projects/anomalib/badge/?version=latest)](https://anomalib.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/openvinotoolkit/anomalib/branch/main/graph/badge.svg?token=Z6A07N1BZK)](https://codecov.io/gh/openvinotoolkit/anomalib)
[![Downloads](https://static.pepy.tech/personalized-badge/anomalib?period=total&units=international_system&left_color=grey&right_color=green&left_text=PyPI%20Downloads)](https://pepy.tech/project/anomalib)
[![Discord](https://img.shields.io/discord/1230798452577800237?style=plastic)](https://discord.com/channels/1230798452577800237)

</div>

---

# 👋 Introduction

Anomalib is a deep learning library that aims to collect state-of-the-art anomaly detection algorithms for benchmarking on both public and private datasets. Anomalib provides several ready-to-use implementations of anomaly detection algorithms described in the recent literature, as well as a set of tools that facilitate the development and implementation of custom models. The library has a strong focus on visual anomaly detection, where the goal of the algorithm is to detect and/or localize anomalies within images or videos in a dataset. Anomalib is constantly updated with new algorithms and training/inference extensions, so keep checking!

<p align="center">
  <img src="https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/_static/images/readme.png" width="1000" alt="A prediction made by anomalib">
</p>

## Key features

- Simple and modular API and CLI for training, inference, benchmarking, and hyperparameter optimization.
- The largest public collection of ready-to-use deep learning anomaly detection algorithms and benchmark datasets.
- [**Lightning**](https://www.lightning.ai/) based model implementations to reduce boilerplate code and limit the implementation efforts to the bare essentials.
- The majority of models can be exported to [**OpenVINO**](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) Intermediate Representation (IR) for accelerated inference on Intel hardware.
- A set of [inference tools](tools) for quick and easy deployment of the standard or custom anomaly detection models.

# 📦 Installation

Anomalib provides two ways to install the library. The first is through PyPI, and the second is through a local installation. PyPI installation is recommended if you want to use the library without making any changes to the source code. If you want to make changes to the library, then a local installation is recommended.

<details>
<summary>Install from PyPI</summary>
Installing the library with pip is the easiest way to get started with anomalib.

```bash
pip install anomalib
```

This will install Anomalib CLI using the [dependencies](/pyproject.toml) in the `pyproject.toml` file. Anomalib CLI is a command line interface for training, inference, benchmarking, and hyperparameter optimization. If you want to use the library as a Python package, you can install the library with the following command:

```bash
# Get help for the installation arguments
anomalib install -h

# Install the full package
anomalib install

# Install with verbose output
anomalib install -v

# Install the core package option only to train and evaluate models via Torch and Lightning
anomalib install --option core

# Install with OpenVINO option only. This is useful for edge deployment as the wheel size is smaller.
anomalib install --option openvino
```

</details>

<details>
<summary>Install from source</summary>
To install from source, you need to clone the repository and install the library using pip via editable mode.

```bash
# Use of virtual environment is highly recommended
# Using conda
yes | conda create -n anomalib_env python=3.10
conda activate anomalib_env

# Or using your favorite virtual environment
# ...

# Clone the repository and install in editable mode
git clone https://github.com/openvinotoolkit/anomalib.git
cd anomalib
pip install -e .
```

This will install Anomalib CLI using the [dependencies](/pyproject.toml) in the `pyproject.toml` file. Anomalib CLI is a command line interface for training, inference, benchmarking, and hyperparameter optimization. If you want to use the library as a Python package, you can install the library with the following command:

```bash
# Get help for the installation arguments
anomalib install -h

# Install the full package
anomalib install

# Install with verbose output
anomalib install -v

# Install the core package option only to train and evaluate models via Torch and Lightning
anomalib install --option core

# Install with OpenVINO option only. This is useful for edge deployment as the wheel size is smaller.
anomalib install --option openvino
```

</details>

# 🧠 Training

Anomalib supports both API and CLI-based training. The API is more flexible and allows for more customization, while the CLI training utilizes command line interfaces, and might be easier for those who would like to use anomalib off-the-shelf.

<details>
<summary>Training via API</summary>

```python
# Import the required modules
from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine

# Initialize the datamodule, model and engine
datamodule = MVTec()
model = Patchcore()
engine = Engine()

# Train the model
engine.fit(datamodule=datamodule, model=model)
```

</details>

<details>
<summary>Training via CLI</summary>

```bash
# Get help about the training arguments, run:
anomalib train -h

# Train by using the default values.
anomalib train --model Patchcore --data anomalib.data.MVTec

# Train by overriding arguments.
anomalib train --model Patchcore --data anomalib.data.MVTec --data.category transistor

# Train by using a config file.
anomalib train --config <path/to/config>
```

</details>

# 🤖 Inference

Anomalib includes multiple inferencing scripts, including Torch, Lightning, Gradio, and OpenVINO inferencers to perform inference using the trained/exported model. Here we show an inference example using the Lightning inferencer. For other inferencers, please refer to the [Inference Documentation](https://anomalib.readthedocs.io).

<details>
<summary>Inference via API</summary>

The following example demonstrates how to perform Lightning inference by loading a model from a checkpoint file.

```python
# Assuming the datamodule, model and engine is initialized from the previous step,
# a prediction via a checkpoint file can be performed as follows:
predictions = engine.predict(
    datamodule=datamodule,
    model=model,
    ckpt_path="path/to/checkpoint.ckpt",
)
```

</details>

<details>
<summary>Inference via CLI</summary>

```bash
# To get help about the arguments, run:
anomalib predict -h

# Predict by using the default values.
anomalib predict --model anomalib.models.Patchcore \
                 --data anomalib.data.MVTec \
                 --ckpt_path <path/to/model.ckpt>

# Predict by overriding arguments.
anomalib predict --model anomalib.models.Patchcore \
                 --data anomalib.data.MVTec \
                 --ckpt_path <path/to/model.ckpt>
                 --return_predictions

# Predict by using a config file.
anomalib predict --config <path/to/config> --return_predictions
```

</details>

# ⚙️ Hyperparameter Optimization

Anomalib supports hyperparameter optimization (HPO) using [wandb](https://wandb.ai/) and [comet.ml](https://www.comet.com/). For more details refer the [HPO Documentation](https://openvinotoolkit.github.io/anomalib/tutorials/hyperparameter_optimization.html)

<details>
<summary>HPO via API</summary>

```python
# To be enabled in v1.1
```

</details>

<details>
<summary>HPO via CLI</summary>

The following example demonstrates how to perform HPO for the Patchcore model.

```bash
anomalib hpo --backend WANDB  --sweep_config tools/hpo/configs/wandb.yaml
```

</details>

# 🧪 Experiment Management

Anomalib is integrated with various libraries for experiment tracking such as Comet, tensorboard, and wandb through [pytorch lighting loggers](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html). For more information, refer to the [Logging Documentation](https://openvinotoolkit.github.io/anomalib/tutorials/logging.html)

<details>
<summary>Experiment Management via API</summary>

```python
# To be enabled in v1.1
```

</details>

<details>
<summary>Experiment Management via CLI</summary>

Below is an example of how to enable logging for hyper-parameters, metrics, model graphs, and predictions on images in the test data-set.

You first need to modify the `config.yaml` file to enable logging. The following example shows how to enable logging:

```yaml
# Place the experiment management config here.
```

```bash
# Place the Experiment Management CLI command here.
```

</details>

# 📊 Benchmarking

Anomalib provides a benchmarking tool to evaluate the performance of the anomaly detection models on a given dataset. The benchmarking tool can be used to evaluate the performance of the models on a given dataset, or to compare the performance of multiple models on a given dataset.

Each model in anomalib is benchmarked on a set of datasets, and the results are available in `src/anomalib/models/<type>/<model_name>/README.md`. For example, the MVTec AD results for the Patchcore model are available in the corresponding [README.md](src/anomalib/models/image/patchcore/README.md#mvtec-ad-dataset) file.

<details>
<summary>Benchmarking via API</summary>

```python
# To be enabled in v1.1
```

</details>

<details>
<summary>Benchmarking via CLI</summary>

To run the benchmarking tool, run the following command:

```bash
anomalib benchmark --config tools/benchmarking/benchmark_params.yaml
```

</details>

# ✍️ Reference

If you use this library and love it, use this to cite it 🤗

```tex
@inproceedings{akcay2022anomalib,
  title={Anomalib: A deep learning library for anomaly detection},
  author={Akcay, Samet and Ameln, Dick and Vaidya, Ashwin and Lakshmanan, Barath and Ahuja, Nilesh and Genc, Utku},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={1706--1710},
  year={2022},
  organization={IEEE}
}
```

# 👥 Contributing

For those who would like to contribute to the library, see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

Thank you to all of the people who have already made a contribution - we appreciate your support!

<a href="https://github.com/openvinotoolkit/anomalib/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openvinotoolkit/anomalib" alt="Contributors to openvinotoolkit/anomalib" />
</a>
