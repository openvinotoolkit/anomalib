# PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization

This is the implementation of the [PaDiM](https://arxiv.org/pdf/2011.08785.pdf) paper.

Model Type: Segmentation

## Description

PaDiM is a patch based algorithm. It relies on a pre-trained CNN feature extractor. The image is broken into patches and embeddings are extracted from each patch using different layers of the feature extractors. The activation vectors from different layers are concatenated to get embedding vectors carrying information from different semantic levels and resolutions. This helps encode fine grained and global contexts. However, since the generated embedding vectors may carry redundant information, dimensions are reduced using random selection. A multivariate gaussian distribution is generated for each patch embedding across the entire training batch. Thus, for each patch of the set of training images, we have a different multivariate gaussian distribution. These gaussian distributions are represented as a matrix of gaussian parameters.

During inference, Mahalanobis distance is used to score each patch position of the test image. It uses the inverse of the covariance matrix calculated for the patch during training. The matrix of Mahalanobis distances forms the anomaly map with higher scores indicating anomalous regions.

## Architecture

![PaDiM Architecture](../../../docs/source/images/padim/architecture.jpg "PaDiM Architecture")

## Usage

`python tools/train.py --model padim`

## Benchmark

All results gathered with seed `42`.

## [MVTec Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Image-Level AUC

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18      | 0.891 | 0.945  | 0.857 |  0.982  | 0.950 | 0.976 | 0.994  | 0.844 |  0.901  |  0.750   |   0.961   | 0.863 | 0.759 |   0.889    |   0.920    | 0.780  |
| Wide ResNet-50 | 0.950 | 0.995  | 0.942 |   1.0   | 0.974 | 0.993 | 0.999  | 0.878 |  0.927  |  0.964   |   0.989   | 0.939 | 0.845 |   0.942    |   0.976    | 0.882  |

### Pixel-Level AUC

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18      | 0.968 | 0.984  | 0.918 |  0.994  | 0.934 | 0.947 | 0.983  | 0.965 |  0.984  |  0.978   |   0.970   | 0.957 | 0.978 |   0.988    |   0.968    | 0.979  |
| Wide ResNet-50 | 0.979 | 0.991  | 0.970 |  0.993  | 0.955 | 0.957 | 0.985  | 0.970 |  0.988  |  0.985   |   0.982   | 0.966 | 0.988 |   0.991    |   0.976    | 0.986  |

### Image F1 Score

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18      | 0.916 | 0.930  | 0.893 |  0.984  | 0.934 | 0.952 | 0.976  | 0.858 |  0.960  |  0.836   |   0.974   | 0.932 | 0.879 |   0.923    |   0.796    | 0.915  |
| Wide ResNet-50 | 0.951 | 0.989  | 0.930 |   1.0   | 0.960 | 0.983 | 0.992  | 0.856 |  0.982  |  0.937   |   0.978   | 0.946 | 0.895 |   0.952    |   0.914    | 0.947  |

### Sample Results

![Sample Result 1](../../../docs/source/images/padim/results/0.png "Sample Result 1")

![Sample Result 2](../../../docs/source/images/padim/results/1.png "Sample Result 2")

![Sample Result 3](../../../docs/source/images/padim/results/2.png "Sample Result 3")
