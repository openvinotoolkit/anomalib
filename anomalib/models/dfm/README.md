# Probabilistic Modeling of Deep Features for Out-of-Distribution and Adversarial Detection

This is the implementation of [DFM](https://arxiv.org/pdf/1909.11786.pdf) paper.

Model Type: Classification

## Description

Fast anomaly classification algorithm that consists of a deep feature extraction stage followed by anomaly classification stage consisting of PCA and class-conditional Gaussian Density Estimation.

### Feature Extraction

Features are extracted by feeding the images through a ResNet18 backbone, which was pre-trained on ImageNet. The output of the penultimate layer (average pooling layer) of the network is used to obtain a semantic feature vector with a fixed length of 2048.

### Anomaly Detection

In the anomaly classification stage, class-conditional PCA transformations and Gaussian Density models are learned. Two types of scores are calculated (i) Feature-reconstruction scores (norm of the difference between the high-dimensional pre-image of a reduced dimension feature and the original high-dimensional feature), and (ii) Negative log-likelihood under the learnt density models. Either of these scores can be used for anomaly detection.

## Usage

`python tools/train.py --model dfm`

## Benchmark

All results gathered with seed `42`.

## [MVTec Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Image-Level AUC

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18      | 0.894 | 0.864  | 0.558 |  0.945  | 0.984 | 0.946 | 0.994  | 0.913 |  0.871  |  0.979   |   0.941   | 0.838 | 0.761 |    0.95    |   0.911    | 0.949  |
| Wide ResNet-50 | 0.891 | 0.978  | 0.540 |  0.979  | 0.977 | 0.974 | 0.990  | 0.891 |  0.931  |  0.947   |   0.839   | 0.809 | 0.700 |   0.911    |   0.915    | 0.981  |

### Image F1 Score

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18      | 0.919 | 0.895  | 0.844 |  0.926  | 0.971 | 0.948 | 0.977  | 0.874 |  0.935  |  0.957   |   0.958   | 0.921 | 0.874 |   0.933    |   0.833    | 0.943  |
| Wide ResNet-50 | 0.951 | 0.960  | 0.844 |  0.990  | 0.970 | 0.959 | 0.976  | 0.848 |  0.944  |  0.913   |   0.912   | 0.919 | 0.859 |   0.893    |   0.815    | 0.961  |
