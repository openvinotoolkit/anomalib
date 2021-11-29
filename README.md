# Anomalib
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)]()
[![python](https://img.shields.io/badge/python-3.6%2B-green)]()
[![pytorch](https://img.shields.io/badge/pytorch-1.7.1%2B-orange)]()
<<<<<<< HEAD
[![openvino](https://img.shields.io/badge/openvino-2021.4-purple)]()
![example branch parameter](https://github.com/openvinotoolkit/anomalib/actions/workflows/tox.yml/badge.svg?branch=development)
=======
[![openvino](https://img.shields.io/badge/openvino-2021.4-blue)]()
[![pipeline status](https://gitlab-icv.inn.intel.com/algo_rnd_team/anomaly/badges/development/pipeline.svg)](https://gitlab-icv.inn.intel.com/algo_rnd_team/anomaly/-/commits/development)
>>>>>>> 35a97d4 (Initial commit)
[![coverage report](https://gitlab-icv.inn.intel.com/algo_rnd_team/anomaly/badges/development/coverage.svg)](https://gitlab-icv.inn.intel.com/algo_rnd_team/anomaly/-/commits/development)


This repository contains state-of-the art anomaly detection algorithms trained and evaluated on both public and private
benchmark datasets. The repo is constantly updated with new algorithms, so keep checking.

## Installation
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
*  [`stfpm`](https://gitlab-icv.inn.intel.com/algo_rnd_team/anomaly/tree/samet/stfpm/anomalib/models/stfpm)



## Benchmark

### [MVTec Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Image-Level AUC
| Model | Backbone    |    Avg    | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| ----- | ----------- | :-------: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| PADIM | ResNet-18   |   0.909   | 0.988  | 0.923 |  0.985  | 0.940 | 0.984 | 0.994  | 0.871 |  0.874  |  0.796   |   0.974   | 0.872 | 0.779 |   0.939    |   0.954    | 0.761  |
| PADIM | Wide ResNet | **0.965** | 0.998  | 0.957 |  0.999  | 0.983 | 0.993 | 0.999  | 0.898 |  0.907  |    -     |   0.992   | 0.951 |   -   |   0.981    |   0.973    | 0.909  |
| DFKDE | ResNet-18   |   0.779   | 0.650  | 0.403 |  0.977  | 0.972 | 0.954 | 0.940  | 0.749 |  0.766  |  0.806   |   0.623   | 0.672 | 0.677 |   0.797    |   0.813    | 0.879  |

### Pixel-Level AUC
| Model | Backbone    |    Avg    | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| ----- | ----------- | :-------: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| PADIM | ResNet-18   |   0.964   | 0.986  | 0.919 |  0.992  | 0.916 | 0.937 | 0.980  | 0.957 |  0.980  |  0.972   |   0.957   | 0.951 | 0.973 |   0.986    |   0.968    | 0.980  |
| PADIM | Wide ResNet | **0.974** | 0.990  | 0.970 |  0.991  | 0.940 | 0.954 | 0.982  | 0.963 |  0.985  |    -     |   0.974   | 0.961 |   -   |   0.988    |   0.973    | 0.986  |
| STFPM | ResNet-18   |   0.961   | 0.984  | 0.988 |  0.982  | 0.957 | 0.940 | 0.981  | 0.940 |  0.974  |  0.983   |   0.968   | 0.973 | 0.983 |   0.984    |   0.800    | 0.983  |
