# PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization

This is the implementation of the [PaDiM](https://arxiv.org/pdf/2011.08785.pdf) paper.

Model Type: Segmentation

## Description

PaDiM is a patch based algorithm. It relies on a pre-trained CNN feature extractor. The image is broken into patches and embeddings are extracted from each patch using different layers of the feature extractors. The activation vectors from different layers are concatenated to get embedding vectors carrying information from different semantic levels and resolutions. This helps encode fine grained and global contexts. However, since the generated embedding vectors may carry redundant information, dimensions are reduced using random selection. A multivariate gaussian distribution is generated for each patch embedding across the entire training batch. Thus, for each patch of the set of training images, we have a different multivariate gaussian distribution. These gaussian distributions are represented as a matrix of gaussian parameters.

During inference, Mahalanobis distance is used to score each patch position of the test image. It uses the inverse of the covariance matrix calculated for the patch during training. The matrix of Mahalanobis distances forms the anomaly map with higher scores indicating anomalous regions.

## Architecture

![PaDiM Architecture](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/padim/architecture.jpg "PaDiM Architecture")

## Usage

`python tools/train.py --model padim`

## Benchmark

All results gathered with seed `42`.

## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

| Model      | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| ---------- | ----------- | -------- | ----------- | ----------- |
| Bottle     | 0.994       | 0.976    | 0.983       | 0.951       |
| Cable      | 0.843       | 0.859    | 0.965       | 0.904       |
| Capsule    | 0.902       | 0.960    | 0.984       | 0.917       |
| Carpet     | 0.945       | 0.930    | 0.984       | 0.949       |
| Grid       | 0.857       | 0.893    | 0.918       | 0.809       |
| Hazelnut   | 0.751       | 0.836    | 0.978       | 0.941       |
| Leather    | 0.982       | 0.984    | 0.994       | 0.983       |
| Metal_nut  | 0.961       | 0.974    | 0.970       | 0.914       |
| Pill       | 0.863       | 0.933    | 0.957       | 0.938       |
| Screw      | 0.759       | 0.879    | 0.978       | 0.923       |
| Tile       | 0.950       | 0.934    | 0.934       | 0.817       |
| Toothbrush | 0.889       | 0.923    | 0.988       | 0.933       |
| Transistor | 0.920       | 0.796    | 0.968       | 0.915       |
| Wood       | 0.976       | 0.952    | 0.948       | 0.923       |
| Zipper     | 0.780       | 0.915    | 0.979       | 0.928       |
| Average    | 0.892       | 0.916    | 0.968       | 0.916       |

## [BTAD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

| Model   | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| ------- | ----------- | -------- | ----------- | ----------- |
| 01      | 0.995       | 0.980    | 0.965       | 0.758       |
| 02      | 0.861       | 0.930    | 0.961       | 0.615       |
| 03      | 0.977       | 0.771    | 0.995       | 0.983       |
| Average | 0.944       | 0.894    | 0.974       | 0.785       |

## [Visa Dataset](https://github.com/amazon-science/spot-diff)

| Model      | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| ---------- | ----------- | -------- | ----------- | ----------- |
| candle     | 0.862       | 0.839    | 0.977       | 0.928       |
| capsules   | 0.609       | 0.769    | 0.926       | 0.564       |
| cashew     | 0.885       | 0.855    | 0.978       | 0.834       |
| chewinggum | 0.981       | 0.975    | 0.989       | 0.842       |
| fryum      | 0.858       | 0.879    | 0.958       | 0.762       |
| macaroni1  | 0.781       | 0.761    | 0.980       | 0.888       |
| macaroni2  | 0.719       | 0.721    | 0.960       | 0.753       |
| pcb1       | 0.872       | 0.827    | 0.987       | 0.878       |
| pcb2       | 0.788       | 0.744    | 0.980       | 0.837       |
| pcb3       | 0.715       | 0.702    | 0.980       | 0.791       |
| pcb4       | 0.968       | 0.946    | 0.968       | 0.800       |
| pipe_fryum | 0.914       | 0.898    | 0.992       | 0.879       |
| Average    | 0.830       | 0.826    | 0.973       | 0.813       |

### Sample Results

![Sample Result 1](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/padim/results/0.png "Sample Result 1")

![Sample Result 2](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/padim/results/1.png "Sample Result 2")

![Sample Result 3](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/padim/results/2.png "Sample Result 3")
