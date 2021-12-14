# PatchCore

Towards Total Recall in Industrial Anomaly Detection

This is the implementation of the [PatchCore](https://arxiv.org/pdf/2106.08265.pdf) paper.

Model Type: Segmentation

## Description

## Architecture

![STFPM Architecture](../../../docs/source/images/patchcore/architecture.jpg "PatchCore Architecture")

## Usage

`python tools/train.py --model patchcore`

## Benchmark

All results gathered with seed `42`.

## [MVTec Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Image-Level AUC

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18      |       | 0.947  | 0.722 |  0.997  | 0.982 | 0.988 | 0.972  | 0.810 |  0.586  |  0.981   |   0.631   | 0.780 | 0.482 |   0.827    |   0.733    | 0.844  |
| Wide ResNet-50 |       | 0.981  | 0.842 |   1.0   | 0.991 | 0.991 | 0.985  | 0.868 |  0.763  |  0.988   |   0.914   | 0.769 | 0.427 |   0.806    |   0.878    | 0.958  |

### Pixel-Level AUC

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18      |       | 0.979  | 0.843 |  0.989  | 0.934 | 0.925 | 0.956  | 0.923 |  0.942  |  0.967   |   0.913   | 0.931 | 0.924 |   0.958    |   0.881    | 0.954  |
| Wide ResNet-50 |       | 0.988  | 0.903 |  0.990  | 0.957 | 0.936 | 0.972  | 0.950 |  0.968  |  0.974   |   0.960   | 0.948 | 0.917 |   0.969    |   0.913    | 0.976  |

### Balanced Accuracy Score

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18      |       | 0.859  | 0.548 |  0.995  | 0.937 | 0.983 | 0.960  | 0.796 |  0.539  |  0.959   |   0.545   |  0.5  |  0.5  |    0.65    |   0.688    | 0.650  |
| Wide ResNet-50 |       | 0.912  | 0.787 |   1.0   | 0.979 | 0.939 | 0.967  | 0.732 |  0.522  |  0.955   |   0.907   | 0.538 |  0.5  |   0.675    |   0.867    | 0.920  |

### Image F1 Score

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18      |       | 0.933  | 0.857 |  0.995  | 0.964 | 0.983 | 0.959  | 0.790 |  0.908  |  0.964   |   0.903   | 0.916 | 0.853 |   0.866    |   0.653    | 0.898  |
| Wide ResNet-50 |       | 0.961  | 0.875 |   1.0   | 0.989 | 0.975 | 0.984  | 0.832 |  0.908  |  0.972   |   0.920   | 0.922 | 0.853 |   0.862    |   0.842    | 0.953  |
