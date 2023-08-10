<div align="center">

<img src="https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/logos/anomalib-wide-blue.png" width="600px">

**A library for benchmarking, developing and deploying deep learning anomaly detection algorithms**

---

[Key Features](#key-features) •
[Getting Started](#getting-started) •
[Docs](https://openvinotoolkit.github.io/anomalib) •
[License](https://github.com/openvinotoolkit/anomalib/blob/main/LICENSE)

[![python](https://img.shields.io/badge/python-3.7%2B-green)]()
[![pytorch](https://img.shields.io/badge/pytorch-1.8.1%2B-orange)]()
[![openvino](https://img.shields.io/badge/openvino-2021.4.2-purple)]()
[![comet](https://custom-icon-badges.herokuapp.com/badge/comet__ml-3.31.7-orange?logo=logo_comet_ml)](https://www.comet.com/site/products/ml-experiment-tracking/?utm_source=anomalib&utm_medium=referral)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/684927c1c76c4c5e94bb53480812fbbb)](https://www.codacy.com/gh/openvinotoolkit/anomalib/dashboard?utm_source=github.com&utm_medium=referral&utm_content=openvinotoolkit/anomalib&utm_campaign=Badge_Grade)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)]()
[![Nightly-Regression Test](https://github.com/openvinotoolkit/anomalib/actions/workflows/nightly.yml/badge.svg)](https://github.com/openvinotoolkit/anomalib/actions/workflows/nightly.yml)
[![Pre-Merge Checks](https://github.com/openvinotoolkit/anomalib/actions/workflows/pre_merge.yml/badge.svg)](https://github.com/openvinotoolkit/anomalib/actions/workflows/pre_merge.yml)
[![codecov](https://codecov.io/gh/openvinotoolkit/anomalib/branch/main/graph/badge.svg?token=Z6A07N1BZK)](https://codecov.io/gh/openvinotoolkit/anomalib)
[![Docs](https://github.com/openvinotoolkit/anomalib/actions/workflows/docs.yml/badge.svg)](https://github.com/openvinotoolkit/anomalib/actions/workflows/docs.yml)
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
- A set of [inference tools](#inference) for quick and easy deployment of the standard or custom anomaly detection models.

---

# Getting Started

Following is a guide on how to get started with `anomalib`. For more details, look at the [Documentation](https://openvinotoolkit.github.io/anomalib).

## Jupyter Notebooks

For getting started with a Jupyter Notebook, please refer to the [Notebooks](notebooks) folder of this repository. Additionally, you can refer to a few created by the community:

## PyPI Install

You can get started with `anomalib` by just using pip.

```bash
pip install anomalib
```

## Local Install

It is highly recommended to use virtual environment when installing anomalib. For instance, with [anaconda](https://www.anaconda.com/products/individual), `anomalib` could be installed as,

```bash
yes | conda create -n anomalib_env python=3.10
conda activate anomalib_env
git clone https://github.com/openvinotoolkit/anomalib.git
cd anomalib
pip install -e .
```

# Training

By default [`python tools/train.py`](tools/train.py)
runs [PADIM](https://arxiv.org/abs/2011.08785) model on `leather` category from the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) [(CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) dataset.

```bash
python tools/train.py    # Train PADIM on MVTec AD leather
```

Training a model on a specific dataset and category requires further configuration. Each model has its own configuration
file, [`config.yaml`](src/anomalib/models/padim/config.yaml)
, which contains data, model and training configurable parameters. To train a specific model on a specific dataset and
category, the config file is to be provided:

```bash
python tools/train.py --config <path/to/model/config.yaml>
```

For example, to train [PADIM](src/anomalib/models/padim) you can use

```bash
python tools/train.py --config src/anomalib/models/padim/config.yaml
```

Alternatively, a model name could also be provided as an argument, where the scripts automatically finds the corresponding config file.

```bash
python tools/train.py --model padim
```

where the currently available models are:

- [CFA](src/anomalib/models/cfa)
- [CFlow](src/anomalib/models/cflow)
- [DFKDE](src/anomalib/models/dfkde)
- [DFM](src/anomalib/models/dfm)
- [DRAEM](src/anomalib/models/draem)
- [EfficientAd](src/anomalib/models/efficient_ad)
- [FastFlow](src/anomalib/models/fastflow)
- [GANomaly](src/anomalib/models/ganomaly)
- [PADIM](src/anomalib/models/padim)
- [PatchCore](src/anomalib/models/patchcore)
- [Reverse Distillation](src/anomalib/models/reverse_distillation)
- [STFPM](src/anomalib/models/stfpm)

## Feature extraction & (pre-trained) backbones

The pre-trained backbones come from [PyTorch Image Models (timm)](https://github.com/rwightman/pytorch-image-models), which are wrapped by `FeatureExtractor`.

For more information, please check our documentation or the [section about feature extraction in "Getting Started with PyTorch Image Models (timm): A Practitioner’s Guide"](https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055#b83b:~:text=ready%20to%20train!-,Feature%20Extraction,-timm%20models%20also>).

Tips:

- Papers With Code has an interface to easily browse models available in timm: [https://paperswithcode.com/lib/timm](https://paperswithcode.com/lib/timm)

- You can also find them with the function `timm.list_models("resnet*", pretrained=True)`

The backbone can be set in the config file, two examples below.

```yaml
model:
  name: cflow
  backbone: wide_resnet50_2
  pre_trained: true
```

## Custom Dataset

It is also possible to train on a custom folder dataset. To do so, `data` section in `config.yaml` is to be modified as follows:

```yaml
dataset:
  name: <name-of-the-dataset>
  format: folder
  path: <path/to/folder/dataset>
  normal_dir: normal # name of the folder containing normal images.
  abnormal_dir: abnormal # name of the folder containing abnormal images.
  normal_test_dir: null # name of the folder containing normal test images.
  task: segmentation # classification or segmentation
  mask: <path/to/mask/annotations> #optional
  extensions: null
  split_ratio: 0.2 # ratio of the normal images that will be used to create a test split
  image_size: 256
  train_batch_size: 32
  test_batch_size: 32
  num_workers: 8
  transform_config:
    train: null
    val: null
  create_validation_set: true
  tiling:
    apply: false
    tile_size: null
    stride: null
    remove_border_count: 0
    use_random_tiling: False
    random_tile_count: 16
```

# Inference

Anomalib includes multiple tools, including Lightning, Gradio, and OpenVINO inferencers, for performing inference with a trained model.

The following command can be used to run PyTorch Lightning inference from the command line:

```bash
python tools/inference/lightning_inference.py -h
```

As a quick example:

```bash
python tools/inference/lightning_inference.py \
    --config src/anomalib/models/padim/config.yaml \
    --weights results/padim/mvtec/bottle/run/weights/model.ckpt \
    --input datasets/MVTec/bottle/test/broken_large/000.png \
    --output results/padim/mvtec/bottle/images
```

Example OpenVINO Inference:

```bash
python tools/inference/openvino_inference.py \
    --weights results/padim/mvtec/bottle/run/openvino/model.bin \
    --metadata results/padim/mvtec/bottle/run/openvino/metadata.json \
    --input datasets/MVTec/bottle/test/broken_large/000.png \
    --output results/padim/mvtec/bottle/images
```

> Ensure that you provide path to `metadata.json` if you want the normalization to be applied correctly.

You can also use Gradio Inference to interact with the trained models using a UI. Refer to our [guide](https://openvinotoolkit.github.io/anomalib/tutorials/inference.html#gradio-inference) for more details.

A quick example:

```bash
python tools/inference/gradio_inference.py \
        --weights results/padim/mvtec/bottle/run/weights/model.ckpt
```

## Exporting Model to ONNX or OpenVINO IR

It is possible to export your model to ONNX or OpenVINO IR

If you want to export your PyTorch model to an OpenVINO model, ensure that `export_mode` is set to `"openvino"` in the respective model `config.yaml`.

```yaml
optimization:
  export_mode: "openvino" # options: openvino, onnx
```

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

Anomablib is integrated with various libraries for experiment tracking such as Comet, tensorboard, and wandb through [pytorch lighting loggers](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html).

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

## Image Datasets

`anomalib` supports a number of benchmarking datasets that are commonly used in the literature. The datasets are automatically downloaded and prepared for training and testing. Here are the benchmarking results on the supported datasets:

### [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

| Model                | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| -------------------- | ----------- | -------- | ----------- | ----------- |
| Cfa                  |             |          |             |             |
| CFlow                |             |          |             |             |
| CsFlow               |             |          |             |             |
| Dfkde                |             |          |             |             |
| Dfm                  | 0.948       | 0.945    | 0.966       | 0.870       |
| Draem                |             |          |             |             |
| Efficient AD         | 0.975       | 0.962    | 0.958       | 0.900       |
| FastFlow             | 0.965       | 0.957    | 0.972       | 0.893       |
| Padim                | 0.892       | 0.916    | 0.968       | 0.916       |
| PatchCore            | 0.987       | 0.984    | 0.976       | 0.908       |
| Reverse Distillation |             |          |             |             |
| Stfpm                | 0.904       | 0.931    | 0.955       | 0.881       |

### [BTAD Dataset](https://github.com/pankajmishra000/VT-ADL)

| Model                | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| -------------------- | ----------- | -------- | ----------- | ----------- |
| Cfa                  |             |          |             |             |
| CFlow                |             |          |             |             |
| CsFlow               |             |          |             |             |
| Dfkde                |             |          |             |             |
| Dfm                  | 0.952       | 0.967    | 0.968       | 0.725       |
| Draem                |             |          |             |             |
| Efficient AD         | 0.904       | 0.921    | 0.825       | 0.539       |
| FastFlow             | 0.921       | 0.905    | 0.946       | 0.638       |
| Padim                | 0.944       | 0.894    | 0.974       | 0.785       |
| PatchCore            | 0.930       | 0.963    | 0.969       | 0.721       |
| Reverse Distillation |             |          |             |             |
| Stfpm                | 0.916       | 0.932    | 0.968       | 0.774       |

### [Visa Dataset](https://github.com/amazon-science/spot-diff)

| Model                | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| -------------------- | ----------- | -------- | ----------- | ----------- |
| Cfa                  |             |          |             |             |
| CFlow                |             |          |             |             |
| CsFlow               |             |          |             |             |
| Dfkde                |             |          |             |             |
| Dfm                  | 0.864       | 0.845    | 0.967       | 0.779       |
| Draem                |             |          |             |             |
| Efficient AD         | 0.803       | 0.807    | 0.956       | 0.793       |
| FastFlow             | 0.908       | 0.866    | 0.968       | 0.829       |
| Padim                | 0.830       | 0.826    | 0.973       | 0.813       |
| PatchCore            | 0.912       | 0.883    | 0.980       | 0.852       |
| Reverse Distillation |             |          |             |             |
| Stfpm                | 0.901       | 0.869    | 0.964       | 0.861       |

## Video Datasets

anomalib now supports video datasets. The following datasets and models are supported:

### UCSD Ped1

| Model  | AUC | RBDC | TBDC |
| ------ | --- | ---- | ---- |
| AI-VAD |     |      |      |
| R-KDE  |     |      |      |

### UCSD Ped2

| Model  | AUC | RBDC | TBDC |
| ------ | --- | ---- | ---- |
| AI-VAD |     |      |      |
| R-KDE  |     |      |      |

### Avenue

| Model  | AUC | RBDC | TBDC |
| ------ | --- | ---- | ---- |
| AI-VAD |     |      |      |
| R-KDE  |     |      |      |

### Shanghai Tech

| Model  | AUC | RBDC | TBDC |
| ------ | --- | ---- | ---- |
| AI-VAD |     |      |      |
| R-KDE  |     |      |      |

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
