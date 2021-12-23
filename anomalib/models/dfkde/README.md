# Deep Feature Kernel Density Estimation

Model Type: Classification

## Description

Fast anomaly classification algorithm that consists of a deep feature extraction stage followed by anomaly classification stage consisting of PCA and Gaussian Kernel Density Estimation.

### Feature Extraction

Features are extracted by feeding the images through a ResNet50 backbone, which was pre-trained on ImageNet. The output of the penultimate layer (average pooling layer) of the network is used to obtain a semantic feature vector with a fixed length of 2048.

### Anomaly Detection

In the anomaly classification stage, the features are first reduced to the first 16 principal components. Gaussian Kernel Density is then used to obtain an estimate of the probability density of new examples, based on the collection of training features obtained during the training phase.

## Usage

`python tools/train.py --model dfkde`

## Benchmark

All results gathered with seed `42`.

## [MVTec Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Image-Level AUC

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18      | 0.762 | 0.646  | 0.577 |  0.669  | 0.965 | 0.863 | 0.951  | 0.751 |  0.698  |  0.806   |   0.729   | 0.607 | 0.694 |   0.767    |   0.839    | 0.866  |
| Wide ResNet-50 | 0.774 | 0.708  | 0.422 |  0.905  | 0.959 | 0.903 | 0.936  | 0.746 |  0.853  |  0.736   |   0.687   | 0.749 | 0.574 |   0.697    |   0.843    | 0.892  |

### Image F1 Score

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18      | 0.872 | 0.864  | 0.844 |  0.854  | 0.960 | 0.898 | 0.942  | 0.793 |  0.908  |  0.827   |   0.894   | 0.916 | 0.859 |   0.853    |   0.756    | 0.916  |
| Wide ResNet-50 | 0.875 | 0.907  | 0.844 |  0.905  | 0.945 | 0.914 | 0.946  | 0.790 |  0.914  |  0.817   |   0.894   | 0.922 | 0.855 |   0.845    |   0.722    | 0.910  |
