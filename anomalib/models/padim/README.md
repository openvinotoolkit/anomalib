# PaDiM
PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization

This is the implementation of the [PaDiM](https://arxiv.org/pdf/2011.08785.pdf) paper.

Model Type: Segmentation

## Usage

`python tools/train.py --model padim`

# Benchmark

All results gathered with seed `42`.

## [MVTec Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Image-Level AUC
|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18      |       | 0.945  | 0.857 |  0.982  | 0.950 | 0.976 | 0.994  | 0.844 |  0.901  |  0.750   |   0.961   | 0.863 | 0.759 |   0.889    |   0.920    | 0.780  |
| Wide ResNet-50 |       | 0.995  | 0.942 |   1.0   | 0.974 | 0.993 | 0.999  | 0.878 |  0.927  |  0.964   |   0.989   | 0.939 | 0.845 |   0.942    |   0.976    | 0.882  |

### Pixel-Level AUC
|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18      |       | 0.984  | 0.918 |  0.994  | 0.934 | 0.947 | 0.983  | 0.965 |  0.984  |  0.978   |   0.970   | 0.957 | 0.978 |   0.988    |   0.968    | 0.979  |
| Wide ResNet-50 |       | 0.991  | 0.970 |  0.993  | 0.955 | 0.957 | 0.985  | 0.970 |  0.988  |  0.985   |   0.982   | 0.966 | 0.988 |   0.991    |   0.976    | 0.986  |

### Balanced Accuracy Score
|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18      |       | 0.896  | 0.736 |  0.963  | 0.889 | 0.860 | 0.942  | 0.763 |  0.804  |  0.668   |   0.883   | 0.662 | 0.634 |   0.792    |   0.829    | 0.656  |
| Wide ResNet-50 |       | 0.977  | 0.870 |   1.0   | 0.903 | 0.965 | 0.975  | 0.787 |  0.913  |  0.904   |   0.961   | 0.868 | 0.703 |   0.875    |   0.929    | 0.831  |

### Image F1 Score
|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18      |       | 0.930  | 0.893 |  0.984  | 0.934 | 0.952 | 0.976  | 0.858 |  0.960  |  0.836   |   0.974   | 0.932 | 0.879 |   0.923    |   0.796    | 0.915  |
| Wide ResNet-50 |       | 0.989  | 0.930 |   1.0   | 0.960 | 0.983 | 0.992  | 0.856 |  0.982  |  0.937   |   0.978   | 0.946 | 0.895 |   0.952    |   0.914    | 0.947  |
