<div align="center">

<img src="https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/_static/images/logos/anomalib-wide-blue.png" width="600px" alt="Anomalib Logo - A deep learning library for anomaly detection">

**A library for benchmarking, developing and deploying deep learning anomaly detection algorithms**

---

[Key Features](#key-features) •
[Docs](https://anomalib.readthedocs.io/en/latest/) •
[Notebooks](examples/notebooks) •
[License](LICENSE)

[![python](https://img.shields.io/badge/python-3.10%2B-green)]()
[![pytorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)]()
[![lightning](https://img.shields.io/badge/lightning-2.2%2B-blue)]()
[![openvino](https://img.shields.io/badge/openvino-2024.0%2B-purple)]()

[![Pre-Merge Checks](https://github.com/openvinotoolkit/anomalib/actions/workflows/pre_merge.yml/badge.svg)](https://github.com/openvinotoolkit/anomalib/actions/workflows/pre_merge.yml)
[![codecov](https://codecov.io/gh/openvinotoolkit/anomalib/branch/main/graph/badge.svg?token=Z6A07N1BZK)](https://codecov.io/gh/openvinotoolkit/anomalib)
[![Downloads](https://static.pepy.tech/personalized-badge/anomalib?period=total&units=international_system&left_color=grey&right_color=green&left_text=PyPI%20Downloads)](https://pepy.tech/project/anomalib)

[![ReadTheDocs](https://readthedocs.org/projects/anomalib/badge/?version=latest)](https://anomalib.readthedocs.io/en/latest/?badge=latest)
[![Anomalib - Gurubase docs](https://img.shields.io/badge/Gurubase-Ask%20Anomalib%20Guru-006BFF)](https://gurubase.io/g/anomalib)

<a href="https://trendshift.io/repositories/6030" target="_blank"><img src="https://trendshift.io/api/badge/repositories/6030" alt="openvinotoolkit%2Fanomalib | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

</div>

---

> 🌟 **Announcing v2.0.0 Release!** 🌟
>
> We're excited to announce the release of Anomalib v2.0.0! This version introduces significant improvements and customization options to enhance your anomaly detection workflows. Please be aware that there are several API changes between `v1.2.0` and `v2.0.0`, so please be careful when updating your existing pipelines. Key features include:
>
> - Multi-GPU support
> - New [dataclasses](https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/data/dataclasses.html) for model in- and outputs.
> - Flexible configuration of [model transforms and data augmentations](https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/data/transforms.html).
> - Configurable modules for pre- and post-processing operations via [`Preprocessor`](https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/models/pre_processor.html) and [`Postprocessor`](https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/models/post_processor.html)
> - Customizable model evaluation workflow with new [Metrics API](https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/evaluation/metrics.html) and [`Evaluator`](https://anomalib.readthedocs.io/en/latest/markdown/guides/how_to/evaluation/evaluator.html) module.
> - Configurable module for visualization via `Visualizer` (docs guide: coming soon)
>
> We value your input! Please share feedback via [GitHub Issues](https://github.com/openvinotoolkit/anomalib/issues) or our [Discussions](https://github.com/openvinotoolkit/anomalib/discussions)

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

Anomalib provides multiple installation options to suit your needs. Choose the one that best fits your requirements:

## 🚀 Install from PyPI

```bash
# Basic installation from PyPI
pip install anomalib

# Full installation with all dependencies
pip install anomalib[full]
```

## 🔧 Install from Source

For contributing or customizing the library:

```bash
git clone https://github.com/openvinotoolkit/anomalib.git
cd anomalib
pip install -e .

# Full development installation with all dependencies
pip install -e .[full]
```

# 🧠 Training

Anomalib supports both API and CLI-based training approaches:

## 🔌 Python API

```python
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

# Initialize components
datamodule = MVTecAD()
model = Patchcore()
engine = Engine()

# Train the model
engine.fit(datamodule=datamodule, model=model)
```

## ⌨️ Command Line

```bash
# Train with default settings
anomalib train --model Patchcore --data anomalib.data.MVTecAD

# Train with custom category
anomalib train --model Patchcore --data anomalib.data.MVTecAD --data.category transistor

# Train with config file
anomalib train --config path/to/config.yaml
```

# 🤖 Inference

Anomalib provides multiple inference options including Torch, Lightning, Gradio, and OpenVINO. Here's how to get started:

## 🔌 Python API

```python
# Load model and make predictions
predictions = engine.predict(
    datamodule=datamodule,
    model=model,
    ckpt_path="path/to/checkpoint.ckpt",
)
```

## ⌨️ Command Line

```bash
# Basic prediction
anomalib predict --model anomalib.models.Patchcore \
                 --data anomalib.data.MVTecAD \
                 --ckpt_path path/to/model.ckpt

# Prediction with results
anomalib predict --model anomalib.models.Patchcore \
                 --data anomalib.data.MVTecAD \
                 --ckpt_path path/to/model.ckpt \
                 --return_predictions
```

> 📘 **Note:** For advanced inference options including Gradio and OpenVINO, check our [Inference Documentation](https://anomalib.readthedocs.io).

# Training on Intel GPUs

> [!Note]
> Currently, only single GPU training is supported on Intel GPUs.
> These commands were tested on Arc 750 and Arc 770.

Ensure that you have PyTorch with XPU support installed. For more information, please refer to the [PyTorch XPU documentation](https://pytorch.org/docs/stable/notes/get_start_xpu.html)

## 🔌 API

```python
from anomalib.data import MVTecAD
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import Stfpm

engine = Engine(
    strategy=SingleXPUStrategy(),
    accelerator=XPUAccelerator(),
)
engine.train(Stfpm(), datamodule=MVTecAD())
```

## ⌨️ CLI

```bash
anomalib train --model Padim --data MVTecAD --trainer.accelerator xpu --trainer.strategy xpu_single
```

# ⚙️ Hyperparameter Optimization

Anomalib supports hyperparameter optimization (HPO) using [Weights & Biases](https://wandb.ai/) and [Comet.ml](https://www.comet.com/).

```bash
# Run HPO with Weights & Biases
anomalib hpo --backend WANDB --sweep_config tools/hpo/configs/wandb.yaml
```

> 📘 **Note:** For detailed HPO configuration, check our [HPO Documentation](https://openvinotoolkit.github.io/anomalib/tutorials/hyperparameter_optimization.html).

# 🧪 Experiment Management

Track your experiments with popular logging platforms through [PyTorch Lightning loggers](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html):

- 📊 Weights & Biases
- 📈 Comet.ml
- 📉 TensorBoard

Enable logging in your config file to track:

- Hyperparameters
- Metrics
- Model graphs
- Test predictions

> 📘 **Note:** For logging setup, see our [Logging Documentation](https://openvinotoolkit.github.io/anomalib/tutorials/logging.html).

# 📊 Benchmarking

Evaluate and compare model performance across different datasets:

```bash
# Run benchmarking with default configuration
anomalib benchmark --config tools/benchmarking/benchmark_params.yaml
```

> 💡 **Tip:** Check individual model performance in their respective README files:
>
> - [Patchcore Results](src/anomalib/models/image/patchcore/README.md#mvtec-ad-dataset)
> - [Other Models](src/anomalib/models/)

# ✍️ Reference

If you find Anomalib useful in your research or work, please cite:

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

We welcome contributions! Check out our [Contributing Guide](CONTRIBUTING.md) to get started.

<p align="center">
  <a href="https://github.com/openvinotoolkit/anomalib/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=openvinotoolkit/anomalib" alt="Contributors to openvinotoolkit/anomalib" />
  </a>
</p>

<p align="center">
  <b>Thank you to all our contributors!</b>
</p>
