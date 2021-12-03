<div align="center">

<img src="docs/anomalib.png" width="400px">

**A library for benchmarking, developing and deploying anomaly detection algorithms in PyTorch**
___

[Key Features](#key-features) •
[Getting Started](#getting-started) •
[Docs](https://openvinotoolkit.github.io/anomalib) •
[License](https://github.com/openvinotoolkit/anomalib/blob/development/LICENSE)

[![black](https://img.shields.io/badge/code%20style-black-000000.svg)]()
[![python](https://img.shields.io/badge/python-3.6%2B-green)]()
[![pytorch](https://img.shields.io/badge/pytorch-1.7.1%2B-orange)]()
[![openvino](https://img.shields.io/badge/openvino-2021.4-purple)]()
![example branch parameter](https://github.com/openvinotoolkit/anomalib/actions/workflows/tox.yml/badge.svg?branch=development)

</div>

___

## Introduction
Anomalib is a deep learning library that aims to collect state-of-the-art anomaly detection algorithms for benchmarking on both public and private datasets. Anomalib provides several ready-to-use implementations of anomaly detection algorithms described in the recent literature, as well as a set of tools that facilitate the development and implementation of custom models. The library has a strong focus on image-based anomaly detection, where the goal of the algorithm is to identify anomalous images, or anomalous pixel regions within images in a dataset. Anomalib is constantly updated with new algorithms and training/inference extensions, so keep checking!

##### Key features:
- The biggest collection of ready-to-use deep learning anomaly detection algorithms and public benchmark datasets.
- [**PyTorch Lightning**](https://www.pytorchlightning.ai/) based model implementations to reduce boilerplate code and limit the implementation efforts to the bare essentials.
- All models can be exported to [**OpenVINO**](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) Intermediate Representation (IR) for accelerated inference on intel hardware.
- A set of [inference tools](#inference) for quick and easy deployment of the standard or custom anomaly detection models.

___
## Getting Started
The repo is thoroughly tested based on the following configuration.
*  Ubuntu 20.04
*  NVIDIA GeForce RTX 3090

You will need [Anaconda](https://www.anaconda.com/products/individual) installed on your system before proceeding with the Anomaly Library install.

After downloading the Anomaly Library, extract the files and navigate to the extracted location.

To perform an installation, run the following:
```
yes | conda create -n anomalib python=3.8
conda activate anomalib
pip install -r requirements/requirements.txt
```

## Training

By default [`python tools/train.py`](https://gitlab-icv.inn.intel.com/algo_rnd_team/anomaly/-/blob/development/train.py)
runs [PADIM](https://arxiv.org/abs/2011.08785) model [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad) `leather` dataset.
```
python tools/train.py    # Train PADIM on MVTec leather
```

Training a model on a specific dataset and category requires further configuration. Each model has its own configuration
file, [`config.yaml`](https://gitlab-icv.inn.intel.com/algo_rnd_team/anomaly/-/blob/development/stfpm/anomalib/models/stfpm/config.yaml)
, which contains data, model and training configurable parameters. To train a specific model on a specific dataset and
category, the config file is to be provided:
```
python tools/train.py --model_config_path <path/to/model/config.yaml>
```

Alternatively, a model name could also be provided as an argument, where the scripts automatically finds the corresponding config file.
```
python tools/train.py --model stfpm
```
where the currently available models are:
* [DFKDE](https://github.com/openvinotoolkit/anomalib/tree/development/anomalib/models/dfkde)
* [DFM](https://github.com/openvinotoolkit/anomalib/tree/development/anomalib/models/dfm)
* [PADIM](https://github.com/openvinotoolkit/anomalib/tree/development/anomalib/models/padim)
* [PatchCore](https://github.com/openvinotoolkit/anomalib/tree/development/anomalib/models/patchcore)
* [STFPM](https://github.com/openvinotoolkit/anomalib/tree/development/anomalib/models/stfpm)

## Inference

## Datasets

## Add a custom model

## Contributing
