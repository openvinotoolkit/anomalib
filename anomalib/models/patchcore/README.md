# PatchCore

This is the implementation of the [PatchCore](https://arxiv.org/pdf/2106.08265.pdf) paper.

Model Type: Segmentation

## Description

The PatchCore algorithm is based on the idea that an image can be classified as anomalous as soon as a single patch is anomalous. The input image is tiled. These tiles act as patches which are fed into the neural network. It consists of a single pre-trained network which is used to extract "mid" level features patches. The "mid" level here refers to the feature extraction layer of the neural network model. Lower level features are generally too broad and higher level features are specific to the dataset the model is trained on. The features extracted during training phase are stored in a memory bank of neighbourhood aware patch level features.

During inference this memory bank is coreset subsampled. Coreset subsampling generates a subset which best approximates the structure of the available set and allows for approximate solution finding. This subset helps reduce the search cost associated with nearest neighbour search. The anomaly score is taken as the maximum distance between the test patch in the test patch collection to each respective nearest neighbour.

## Architecture

![PatchCore Architecture](../../../docs/source/images/patchcore/architecture.jpg "PatchCore Architecture")

## Usage

`python tools/train.py --model patchcore`

## Benchmark

All results gathered with seed `42`.

## [MVTec Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Image-Level AUC

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18      | 0.819 | 0.947  | 0.722 |  0.997  | 0.982 | 0.988 | 0.972  | 0.810 |  0.586  |  0.981   |   0.631   | 0.780 | 0.482 |   0.827    |   0.733    | 0.844  |
| Wide ResNet-50 | 0.877 | 0.981  | 0.842 |   1.0   | 0.991 | 0.991 | 0.985  | 0.868 |  0.763  |  0.988   |   0.914   | 0.769 | 0.427 |   0.806    |   0.878    | 0.958  |

### Pixel-Level AUC

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18      | 0.935 | 0.979  | 0.843 |  0.989  | 0.934 | 0.925 | 0.956  | 0.923 |  0.942  |  0.967   |   0.913   | 0.931 | 0.924 |   0.958    |   0.881    | 0.954  |
| Wide ResNet-50 | 0.955 | 0.988  | 0.903 |  0.990  | 0.957 | 0.936 | 0.972  | 0.950 |  0.968  |  0.974   |   0.960   | 0.948 | 0.917 |   0.969    |   0.913    | 0.976  |

### Image F1 Score

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| ResNet-18      | 0.896 | 0.933  | 0.857 |  0.995  | 0.964 | 0.983 | 0.959  | 0.790 |  0.908  |  0.964   |   0.903   | 0.916 | 0.853 |   0.866    |   0.653    | 0.898  |
| Wide ResNet-50 | 0.923 | 0.961  | 0.875 |   1.0   | 0.989 | 0.975 | 0.984  | 0.832 |  0.908  |  0.972   |   0.920   | 0.922 | 0.853 |   0.862    |   0.842    | 0.953  |

### Sample Results

![Sample Result 1](../../../docs/source/images/patchcore/results/0.png "Sample Result 1")

![Sample Result 2](../../../docs/source/images/patchcore/results/1.png "Sample Result 2")

![Sample Result 3](../../../docs/source/images/patchcore/results/2.png "Sample Result 3")
