# Anomalib

This repository contains state-of-the art anomaly detection algorithms trained 
and evaluated on both public and private benchmark datasets. The repo is 
constantly updated with new algorithms, so keep checking.

## Installation
The repo is thoroughly tested based on the following configuration.
*  Ubuntu 20.04
*  NVIDIA GeForce RTX 3090

To perform a development install, run the following:
```
yes | conda create -n anomalib python=3.8
conda activate anomalib
pip install -r requirements.txt
```

## Training
By default [`python train.py`](https://gitlab-icv.inn.intel.com/algo_rnd_team/anomaly/blob/samet/stfpm/train.py)
runs [STFPM](https://arxiv.org/pdf/2103.04257.pdf) model [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad) `leather` dataset.
```
python train.py    # Train STFPM on MVTec leather
```

Training a model on a specific dataset and category requires further configuration. Each model has its own 
configuration file, [`config.yaml`](https://gitlab-icv.inn.intel.com/algo_rnd_team/anomaly/blob/samet/stfpm/anomalib/models/stfpm/config.yaml), which contains data, model and training 
configurable parameters. To train a specific model on a specific dataset and category, the config file is to be provided:
```
python train.py --model_config_path <path/to/model/config.yaml>
```

Alternatively, a model name could also be provided as an argument, where the scripts automatically finds the corresponding config file.
```
python train.py --model stfpm
```
where the currently available models are:
*  [`stfpm`](https://gitlab-icv.inn.intel.com/algo_rnd_team/anomaly/tree/samet/stfpm/anomalib/models/stfpm)



## Benchmark

### [MVTec Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

|       | Avg        | Carpet |  Grid | Leather |  Tile |  Wood | Bottle | Cable | Capsule | Hazelnut | Metal Nut |  Pill | Screw | Toothbrush | Transistor | Zipper |
|-------|:----------:|:------:|:-----:|:-------:|:-----:|:-----:|:------:|:-----:|:-------:|:--------:|:---------:|:-----:|:-----:|:----------:|:----------:|:------:|
| STFPM |  **0.961** |  0.984 | 0.988 |  0.982  | 0.957 | 0.940 |  0.981 | 0.940 |  0.974  |   0.983  |   0.968   | 0.973 | 0.983 |    0.984   |    0.800   |  0.983 |

