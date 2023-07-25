# PatchCore

This is the implementation of the [PatchCore](https://arxiv.org/pdf/2106.08265.pdf) paper.

Model Type: Segmentation

## Description

The PatchCore algorithm is based on the idea that an image can be classified as anomalous as soon as a single patch is anomalous. The input image is tiled. These tiles act as patches which are fed into the neural network. It consists of a single pre-trained network which is used to extract "mid" level features patches. The "mid" level here refers to the feature extraction layer of the neural network model. Lower level features are generally too broad and higher level features are specific to the dataset the model is trained on. The features extracted during training phase are stored in a memory bank of neighbourhood aware patch level features.

During inference this memory bank is coreset subsampled. Coreset subsampling generates a subset which best approximates the structure of the available set and allows for approximate solution finding. This subset helps reduce the search cost associated with nearest neighbour search. The anomaly score is taken as the maximum distance between the test patch in the test patch collection to each respective nearest neighbour.

## Architecture

![PatchCore Architecture](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/patchcore/architecture.jpg "PatchCore Architecture")

## Usage

`python tools/train.py --model patchcore`

## Benchmark

All results gathered with seed `42`.

## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

| Model      | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| ---------- | ----------- | -------- | ----------- | ----------- |
| Bottle     | 1.000       | 1.000    | 0.980       | 0.921       |
| Cable      | 0.987       | 0.978    | 0.980       | 0.904       |
| Capsule    | 0.976       | 0.977    | 0.987       | 0.919       |
| Carpet     | 0.978       | 0.971    | 0.987       | 0.928       |
| Grid       | 0.972       | 0.965    | 0.979       | 0.899       |
| Hazelnut   | 1.000       | 1.000    | 0.984       | 0.943       |
| Leather    | 1.000       | 1.000    | 0.988       | 0.963       |
| Metal_nut  | 0.997       | 0.989    | 0.984       | 0.915       |
| Pill       | 0.937       | 0.953    | 0.974       | 0.937       |
| Screw      | 0.976       | 0.966    | 0.991       | 0.942       |
| Tile       | 0.987       | 0.988    | 0.947       | 0.790       |
| Toothbrush | 1.000       | 1.000    | 0.987       | 0.859       |
| Transistor | 1.000       | 1.000    | 0.965       | 0.938       |
| Wood       | 0.993       | 0.983    | 0.931       | 0.844       |
| Zipper     | 0.994       | 0.987    | 0.980       | 0.922       |
| Average    | 0.987       | 0.984    | 0.976       | 0.908       |

## [BTAD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

| Model   | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| ------- | ----------- | -------- | ----------- | ----------- |
| 01      | 0.976       | 0.970    | 0.962       | 0.660       |
| 02      | 0.815       | 0.930    | 0.950       | 0.525       |
| 03      | 1.000       | 0.988    | 0.995       | 0.978       |
| Average | 0.930       | 0.963    | 0.969       | 0.721       |

## [Visa Dataset](https://github.com/amazon-science/spot-diff)

| Model      | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| ---------- | ----------- | -------- | ----------- | ----------- |
| candle     | 0.987       | 0.950    | 0.989       | 0.941       |
| capsules   | 0.678       | 0.778    | 0.972       | 0.659       |
| cashew     | 0.958       | 0.916    | 0.990       | 0.895       |
| chewinggum | 0.997       | 0.985    | 0.989       | 0.855       |
| fryum      | 0.916       | 0.886    | 0.952       | 0.794       |
| macaroni1  | 0.898       | 0.837    | 0.983       | 0.916       |
| macaroni2  | 0.741       | 0.739    | 0.967       | 0.879       |
| pcb1       | 0.952       | 0.886    | 0.995       | 0.879       |
| pcb2       | 0.926       | 0.868    | 0.978       | 0.828       |
| pcb3       | 0.912       | 0.839    | 0.981       | 0.804       |
| pcb4       | 0.995       | 0.966    | 0.977       | 0.842       |
| pipe_fryum | 0.981       | 0.951    | 0.989       | 0.934       |
| Average    | 0.912       | 0.883    | 0.980       | 0.852       |

### Sample Results

![Sample Result 1](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/patchcore/results/0.png "Sample Result 1")

![Sample Result 2](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/patchcore/results/1.png "Sample Result 2")

![Sample Result 3](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/patchcore/results/2.png "Sample Result 3")
