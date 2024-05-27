# Probabilistic Modeling of Deep Features for Out-of-Distribution and Adversarial Detection

This is the implementation of [DFM](https://arxiv.org/pdf/1909.11786.pdf) paper.

Model Type: Classification

## Description

Fast anomaly classification algorithm that consists of a deep feature extraction stage followed by anomaly classification stage consisting of PCA and class-conditional Gaussian Density Estimation.

### Feature Extraction

Features are extracted by feeding the images through a ResNet18 backbone, which was pre-trained on ImageNet. The output of the penultimate layer (average pooling layer) of the network is used to obtain a semantic feature vector with a fixed length of 2048.

### Anomaly Detection

In the anomaly classification stage, class-conditional PCA transformations and Gaussian Density models are learned. Two types of scores are calculated (i) Feature-reconstruction scores (norm of the difference between the high-dimensional pre-image of a reduced dimension feature and the original high-dimensional feature), and (ii) Negative log-likelihood under the learnt density models. Anomaly map generation is supported only with the feature-reconstruction based scores. Image level anomaly detection is supported by both score types.

## Usage

`anomalib train --model Dfm --data MVTec --data.category <category>`

## Benchmark

All results gathered with seed `42`.

## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

> Note: Metrics for ResNet 18 were calculated with pooling kernel size of 2 while for Wide ResNet 50, kernel size of 4 was used.

### Image-Level AUC

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18      | 0.936 | 0.817  | 0.736 |  0.993  | 0.966 | 0.977 |   1    | 0.956 |  0.944  |  0.994   |   0.922   | 0.961 | 0.89  |   0.969    |   0.939    | 0.969  |
| Wide ResNet-50 | 0.943 | 0.855  | 0.784 |  0.997  | 0.995 | 0.975 | 0.999  | 0.969 |  0.924  |  0.978   |   0.939   | 0.962 | 0.873 |   0.969    |   0.971    | 0.961  |

### Image F1 Score

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :--: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18      | 0.943 | 0.895  | 0.871 |  0.978  | 0.958 | 0.96 |   1    | 0.935 |  0.965  |  0.966   |   0.942   | 0.956 | 0.914 |   0.966    |   0.868    | 0.964  |
| Wide ResNet-50 | 0.950 | 0.915  | 0.87  |  0.995  | 0.988 | 0.96 | 0.992  | 0.939 |  0.965  |  0.971   |   0.942   | 0.956 | 0.906 |   0.966    |   0.914    | 0.971  |
