<div align="center">

<img src="https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/logos/anomalib-wide-blue.png" width="600px">

**A library for benchmarking, developing and deploying deep learning anomaly detection algorithms**

---

[Key Features](#key-features) •
[Getting Started](#getting-started) •
[Docs](https://anomalib.readthedocs.io/en/latest/) •
[License](https://github.com/openvinotoolkit/anomalib/blob/main/LICENSE)

[![python](https://img.shields.io/badge/python-3.7%2B-green)]()
[![pytorch](https://img.shields.io/badge/pytorch-1.8.1%2B-orange)]()
[![openvino](https://img.shields.io/badge/openvino-2022.3.0-purple)]()
<br/>
[![Pre-Merge Checks](https://github.com/openvinotoolkit/anomalib/actions/workflows/pre_merge.yml/badge.svg)](https://github.com/openvinotoolkit/anomalib/actions/workflows/pre_merge.yml)
[![Documentation Status](https://readthedocs.org/projects/anomalib/badge/?version=latest)](https://anomalib.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/openvinotoolkit/anomalib/branch/main/graph/badge.svg?token=Z6A07N1BZK)](https://codecov.io/gh/openvinotoolkit/anomalib)
[![Downloads](https://static.pepy.tech/personalized-badge/anomalib?period=total&units=international_system&left_color=grey&right_color=green&left_text=PyPI%20Downloads)](https://pepy.tech/project/anomalib)

</div>

---

# Introduction

Anomalib is a deep learning library that aims to collect state-of-the-art anomaly detection algorithms for benchmarking on both public and private datasets. Anomalib provides several ready-to-use implementations of anomaly detection algorithms described in the recent literature, as well as a set of tools that facilitate the development and implementation of custom models. The library has a strong focus on image-based anomaly detection, where the goal of the algorithm is to identify anomalous images, or anomalous pixel regions within images in a dataset. Anomalib is constantly updated with new algorithms and training/inference extensions, so keep checking!

![Sample Image](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/readme.png)

## Key features

- The largest public collection of ready-to-use deep learning anomaly detection algorithms and benchmark datasets.
- [**PyTorch Lightning**](https://www.pytorchlightning.ai/) based model implementations to reduce boilerplate code and limit the implementation efforts to the bare essentials.
- All models can be exported to [**OpenVINO**](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) Intermediate Representation (IR) for accelerated inference on intel hardware.
- A set of [inference tools](tools) for quick and easy deployment of the standard or custom anomaly detection models.

---

## Getting Started

Anomalib provides a set of resources to get started with the library. These include:

- Library [Documentation](https://anomalib.readthedocs.io)
- Jupyter [Notebooks](notebooks)

# 📦 Installation

Anomalib provides two ways to install the library. The first is through PyPI, and the second is through a local installation. PyPI installation is recommended if you want to use the library without making any changes to the source code. If you want to make changes to the library, then a local installation is recommended.

<details>
<summary>Install from PyPI</summary>
Installing the library with pip is the easiest way to get started with `anomalib`.

```bash
pip install anomalib
```

</details>

<details>
<summary>Install from source</summary>
To install from source, you need to clone the repository and install the library using pip via editable mode.

```bash
# Use of virtual environment is highy recommended
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

</details>

# 🧠 Training

Anomalib now supports both API and CLI-based training. The API is more flexible and allows for more customization, while the CLI training utilizes command line interfaces, and might be easier for those who would like to use anomalib off-the-shelf.

<details>
<summary>Training via API</summary>

```python
# Import the required modules
from anomalib.datasets import MVTec
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
# To get help about the training arguments, run:
anomalib train -h

# To train the model on the default values, run:
anomalib train --model anomalib.models.patchcore --data anomalib.data.MVTec
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

# Predict via providing the model name
anomalib predict --model anomalib.models.Patchcore

# Predict by overriding the various values with commands.
anomalib predict --model anomalib.models.Patchcore --data <CONFIG | CLASS_PATH_OR_NAME>

# Predict by using a config file.
anomalib predict --config <config_file_path> --return_predictions
```

</details>

# Hyperparameter Optimization

To run hyperparameter optimization, use the following command:

```bash
python tools/hpo/sweep.py \
    --model padim --model_config ./path_to_config.yaml \
    --sweep_config tools/hpo/sweep.yaml
```

For more details refer the [HPO Documentation](https://openvinotoolkit.github.io/anomalib/tutorials/hyperparameter_optimization.html)

# Benchmarking

To gather benchmarking data such as throughput across categories, use the following command:

```bash
python tools/benchmarking/benchmark.py \
    --config <relative/absolute path>/<paramfile>.yaml
```

Refer to the [Benchmarking Documentation](https://openvinotoolkit.github.io/anomalib/tutorials/benchmarking.html) for more details.

# Experiment Management

Anomalib is integrated with various libraries for experiment tracking such as Comet, tensorboard, and wandb through [pytorch lighting loggers](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html).

Below is an example of how to enable logging for hyper-parameters, metrics, model graphs, and predictions on images in the test data-set

```yaml
visualization:
  log_images: True # log images to the available loggers (if any)
  mode: full # options: ["full", "simple"]

 logging:
  logger: [comet, tensorboard, wandb]
  log_graph: True
```

For more information, refer to the [Logging Documentation](https://openvinotoolkit.github.io/anomalib/tutorials/logging.html)

Note: Set your API Key for [Comet.ml](https://www.comet.com/signup?utm_source=anomalib&utm_medium=referral) via `comet_ml.init()` in interactive python or simply run `export COMET_API_KEY=<Your API Key>`

# Community Projects

## 1. Web-based Pipeline for Training and Inference

This project showcases an end-to-end training and inference pipeline build on top of Anomalib. It provides a web-based UI for uploading MVTec style datasets and training them on the available Anomalib models. It also has sections for calling inference on individual images as well as listing all the images with their predictions in the database.

You can view the project on [Github](https://github.com/vnk8071/anomaly-detection-in-industry-manufacturing/tree/master/anomalib_contribute)
For more details see the [Discussion forum](https://github.com/openvinotoolkit/anomalib/discussions/733)

# Datasets

`anomalib` supports MVTec AD [(CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) and BeanTech [(CC-BY-SA)](https://creativecommons.org/licenses/by-sa/4.0/legalcode) for benchmarking and `folder` for custom dataset training/inference.

## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

MVTec AD dataset is one of the main benchmarks for anomaly detection, and is released under the
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License [(CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

> Note: These metrics are collected with image size of 256 and seed `42`. This common setting is used to make model comparisons fair.

## Image-Level AUC

| Model           |                |    Avg    |  Carpet   |   Grid    |  Leather  |   Tile    |   Wood    |  Bottle   |   Cable   |  Capsule  | Hazelnut | Metal Nut |   Pill    |   Screw   | Toothbrush | Transistor |  Zipper   |
| --------------- | -------------- | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :------: | :-------: | :-------: | :-------: | :--------: | :--------: | :-------: |
| **EfficientAd** | **PDN-S**      | **0.982** |   0.982   | **1.000** |   0.997   | **1.000** |   0.986   | **1.000** |   0.952   |   0.950   |  0.952   |   0.979   | **0.987** |   0.960   |   0.997    |   0.999    | **0.994** |
| EfficientAd     | PDN-M          |   0.975   |   0.972   |   0.998   | **1.000** |   0.999   |   0.984   |   0.991   |   0.945   |   0.957   |  0.948   |   0.989   |   0.926   | **0.975** | **1.000**  |   0.965    |   0.971   |
| PatchCore       | Wide ResNet-50 |   0.980   |   0.984   |   0.959   |   1.000   | **1.000** |   0.989   |   1.000   | **0.990** | **0.982** |  1.000   |   0.994   |   0.924   |   0.960   |   0.933    | **1.000**  |   0.982   |
| PatchCore       | ResNet-18      |   0.973   |   0.970   |   0.947   |   1.000   |   0.997   |   0.997   |   1.000   |   0.986   |   0.965   |  1.000   |   0.991   |   0.916   |   0.943   |   0.931    |   0.996    |   0.953   |
| CFlow           | Wide ResNet-50 |   0.962   |   0.986   |   0.962   | **1.000** |   0.999   |   0.993   |  **1.0**  |   0.893   |   0.945   | **1.0**  | **0.995** |   0.924   |   0.908   |   0.897    |   0.943    |   0.984   |
| CFA             | Wide ResNet-50 |   0.956   |   0.978   |   0.961   |   0.990   |   0.999   |   0.994   |   0.998   |   0.979   |   0.872   |  1.000   | **0.995** |   0.946   |   0.703   | **1.000**  |   0.957    |   0.967   |
| CFA             | ResNet-18      |   0.930   |   0.953   |   0.947   |   0.999   |   1.000   | **1.000** |   0.991   |   0.947   |   0.858   |  0.995   |   0.932   |   0.887   |   0.625   |   0.994    |   0.895    |   0.919   |
| PaDiM           | Wide ResNet-50 |   0.950   | **0.995** |   0.942   | **1.000** |   0.974   |   0.993   |   0.999   |   0.878   |   0.927   |  0.964   |   0.989   |   0.939   |   0.845   |   0.942    |   0.976    |   0.882   |
| PaDiM           | ResNet-18      |   0.891   |   0.945   |   0.857   |   0.982   |   0.950   |   0.976   |   0.994   |   0.844   |   0.901   |  0.750   |   0.961   |   0.863   |   0.759   |   0.889    |   0.920    |   0.780   |
| DFM             | Wide ResNet-50 |   0.943   |   0.855   |   0.784   |   0.997   |   0.995   |   0.975   |   0.999   |   0.969   |   0.924   |  0.978   |   0.939   |   0.962   |   0.873   |   0.969    |   0.971    |   0.961   |
| DFM             | ResNet-18      |   0.936   |   0.817   |   0.736   |   0.993   |   0.966   |   0.977   |   1.000   |   0.956   |   0.944   |  0.994   |   0.922   |   0.961   |   0.89    |   0.969    |   0.939    |   0.969   |
| STFPM           | Wide ResNet-50 |   0.876   |   0.957   |   0.977   |   0.981   |   0.976   |   0.939   |   0.987   |   0.878   |   0.732   |  0.995   |   0.973   |   0.652   |   0.825   |   0.500    |   0.875    |   0.899   |
| STFPM           | ResNet-18      |   0.893   |   0.954   | **0.982** |   0.989   |   0.949   |   0.961   |   0.979   |   0.838   |   0.759   |  0.999   |   0.956   |   0.705   |   0.835   | **0.997**  |   0.853    |   0.645   |
| DFKDE           | Wide ResNet-50 |   0.774   |   0.708   |   0.422   |   0.905   |   0.959   |   0.903   |   0.936   |   0.746   |   0.853   |  0.736   |   0.687   |   0.749   |   0.574   |   0.697    |   0.843    |   0.892   |
| DFKDE           | ResNet-18      |   0.762   |   0.646   |   0.577   |   0.669   |   0.965   |   0.863   |   0.951   |   0.751   |   0.698   |  0.806   |   0.729   |   0.607   |   0.694   |   0.767    |   0.839    |   0.866   |
| GANomaly        |                |   0.421   |   0.203   |   0.404   |   0.413   |   0.408   |   0.744   |   0.251   |   0.457   |   0.682   |  0.537   |   0.270   |   0.472   |   0.231   |   0.372    |   0.440    |   0.434   |

## Pixel-Level AUC

| Model       |                    |    Avg    |  Carpet   |   Grid    |  Leather  |   Tile    |   Wood    |  Bottle   |   Cable   |  Capsule  | Hazelnut  | Metal Nut |   Pill    |   Screw   | Toothbrush | Transistor |  Zipper   |
| ----------- | ------------------ | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :--------: | :--------: | :-------: |
| **CFA**     | **Wide ResNet-50** | **0.983** |   0.980   |   0.954   |   0.989   | **0.985** | **0.974** | **0.989** | **0.988** | **0.989** |   0.985   | **0.992** | **0.988** |   0.979   | **0.991**  |   0.977    | **0.990** |
| CFA         | ResNet-18          |   0.979   |   0.970   |   0.973   |   0.992   |   0.978   |   0.964   |   0.986   |   0.984   |   0.987   |   0.987   |   0.981   |   0.981   |   0.973   |   0.990    |   0.964    |   0.978   |
| PatchCore   | Wide ResNet-50     |   0.980   |   0.988   |   0.968   |   0.991   |   0.961   |   0.934   |   0.984   | **0.988** |   0.988   |   0.987   |   0.989   |   0.980   | **0.989** |   0.988    | **0.981**  |   0.983   |
| PatchCore   | ResNet-18          |   0.976   |   0.986   |   0.955   |   0.990   |   0.943   |   0.933   |   0.981   |   0.984   |   0.986   |   0.986   |   0.986   |   0.974   |   0.991   |   0.988    |   0.974    |   0.983   |
| CFlow       | Wide ResNet-50     |   0.971   |   0.986   |   0.968   |   0.993   |   0.968   |   0.924   |   0.981   |   0.955   |   0.988   | **0.990** |   0.982   |   0.983   |   0.979   |   0.985    |   0.897    |   0.980   |
| PaDiM       | Wide ResNet-50     |   0.979   | **0.991** |   0.970   |   0.993   |   0.955   |   0.957   |   0.985   |   0.970   |   0.988   |   0.985   |   0.982   |   0.966   |   0.988   | **0.991**  |   0.976    |   0.986   |
| PaDiM       | ResNet-18          |   0.968   |   0.984   |   0.918   | **0.994** |   0.934   |   0.947   |   0.983   |   0.965   |   0.984   |   0.978   |   0.970   |   0.957   |   0.978   |   0.988    |   0.968    |   0.979   |
| EfficientAd | PDN-S              |   0.960   |   0.963   |   0.937   |   0.976   |   0.907   |   0.868   |   0.983   |   0.983   |   0.980   |   0.976   |   0.978   |   0.986   |   0.985   |   0.962    |   0.956    |   0.961   |
| EfficientAd | PDN-M              |   0.957   |   0.948   |   0.937   |   0.976   |   0.906   |   0.867   |   0.976   |   0.986   |   0.957   |   0.977   |   0.984   |   0.978   |   0.986   |   0.964    |   0.947    |   0.960   |
| STFPM       | Wide ResNet-50     |   0.903   |   0.987   | **0.989** |   0.980   |   0.966   |   0.956   |   0.966   |   0.913   |   0.956   |   0.974   |   0.961   |   0.946   |   0.988   |   0.178    |   0.807    |   0.980   |
| STFPM       | ResNet-18          |   0.951   |   0.986   |   0.988   |   0.991   |   0.946   |   0.949   |   0.971   |   0.898   |   0.962   |   0.981   |   0.942   |   0.878   |   0.983   |   0.983    |   0.838    |   0.972   |

## Image F1 Score

| Model         |                    |    Avg    |  Carpet   |   Grid    |  Leather  |   Tile    |   Wood    |  Bottle   |   Cable   |  Capsule  | Hazelnut  | Metal Nut |   Pill    |   Screw   | Toothbrush | Transistor |  Zipper   |
| ------------- | ------------------ | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :--------: | :--------: | :-------: |
| **PatchCore** | **Wide ResNet-50** | **0.976** |   0.971   |   0.974   | **1.000** | **1.000** |   0.967   | **1.000** |   0.968   | **0.982** | **1.000** |   0.984   |   0.940   |   0.943   |   0.938    | **1.000**  | **0.979** |
| PatchCore     | ResNet-18          |   0.970   |   0.949   |   0.946   | **1.000** |   0.98    |   0.992   | **1.000** | **0.978** |   0.969   | **1.000** | **0.989** |   0.940   |   0.932   |   0.935    |   0.974    |   0.967   |
| EfficientAd   | PDN-S              |   0.970   |   0.966   | **1.000** |   0.995   | **1.000** |   0.975   | **1.000** |   0.907   |   0.956   |   0.897   |   0.978   |   0.982   |   0.944   |   0.984    |   0.988    |   0.983   |
| EfficientAd   | PDN-M              |   0.966   |   0.977   |   0.991   | **1.000** |   0.994   |   0.967   |   0.984   |   0.922   |   0.969   |   0.884   |   0.984   |   0.952   |   0.955   |   1.000    |   0.929    |   0.979   |
| CFA           | Wide ResNet-50     |   0.962   |   0.961   |   0.957   |   0.995   |   0.994   |   0.983   |   0.984   |   0.962   |   0.946   | **1.000** |   0.984   | **0.952** |   0.855   | **1.000**  |   0.907    |   0.975   |
| CFA           | ResNet-18          |   0.946   |   0.956   |   0.946   |   0.973   | **1.000** | **1.000** |   0.983   |   0.907   |   0.938   |   0.996   |   0.958   |   0.920   |   0.858   |   0.984    |   0.795    |   0.949   |
| CFlow         | Wide ResNet-50     |   0.944   |   0.972   |   0.932   | **1.000** |   0.988   |   0.967   | **1.000** |   0.832   |   0.939   | **1.000** |   0.979   |   0.924   | **0.971** |   0.870    |   0.818    |   0.967   |
| PaDiM         | Wide ResNet-50     |   0.951   | **0.989** |   0.930   | **1.000** |   0.960   |   0.983   |   0.992   |   0.856   | **0.982** |   0.937   |   0.978   |   0.946   |   0.895   |   0.952    |   0.914    |   0.947   |
| PaDiM         | ResNet-18          |   0.916   |   0.930   |   0.893   |   0.984   |   0.934   |   0.952   |   0.976   |   0.858   |   0.960   |   0.836   |   0.974   |   0.932   |   0.879   |   0.923    |   0.796    |   0.915   |
| DFM           | Wide ResNet-50     |   0.950   |   0.915   |   0.870   |   0.995   |   0.988   |   0.960   |   0.992   |   0.939   |   0.965   |   0.971   |   0.942   |   0.956   |   0.906   |   0.966    |   0.914    |   0.971   |
| DFM           | ResNet-18          |   0.943   |   0.895   |   0.871   |   0.978   |   0.958   |   0.900   |   1.000   |   0.935   |   0.965   |   0.966   |   0.942   |   0.956   |   0.914   |   0.966    |   0.868    |   0.964   |
| STFPM         | Wide ResNet-50     |   0.926   |   0.973   |   0.973   |   0.974   |   0.965   |   0.929   |   0.976   |   0.853   |   0.920   |   0.972   |   0.974   |   0.922   |   0.884   |   0.833    |   0.815    |   0.931   |
| STFPM         | ResNet-18          |   0.932   |   0.961   | **0.982** |   0.989   |   0.930   |   0.951   |   0.984   |   0.819   |   0.918   |   0.993   |   0.973   |   0.918   |   0.887   | **0.984**  |   0.790    |   0.908   |
| DFKDE         | Wide ResNet-50     |   0.875   |   0.907   |   0.844   |   0.905   |   0.945   |   0.914   |   0.946   |   0.790   |   0.914   |   0.817   |   0.894   |   0.922   |   0.855   |   0.845    |   0.722    |   0.910   |
| DFKDE         | ResNet-18          |   0.872   |   0.864   |   0.844   |   0.854   |   0.960   |   0.898   |   0.942   |   0.793   |   0.908   |   0.827   |   0.894   |   0.916   |   0.859   |   0.853    |   0.756    |   0.916   |
| GANomaly      |                    |   0.834   |   0.864   |   0.844   |   0.852   |   0.836   |   0.863   |   0.863   |   0.760   |   0.905   |   0.777   |   0.894   |   0.916   |   0.853   |   0.833    |   0.571    |   0.881   |

# Reference

If you use this library and love it, use this to cite it 🤗

```tex
@misc{anomalib,
      title={Anomalib: A Deep Learning Library for Anomaly Detection},
      author={Samet Akcay and
              Dick Ameln and
              Ashwin Vaidya and
              Barath Lakshmanan and
              Nilesh Ahuja and
              Utku Genc},
      year={2022},
      eprint={2202.08341},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Contributing

For those who would like to contribute to the library, see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

Thank you to all of the people who have already made a contribution - we appreciate your support!

<a href="https://github.com/openvinotoolkit/anomalib/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openvinotoolkit/anomalib" />
</a>
