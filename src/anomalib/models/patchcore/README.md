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
| Bottle     |             |          |             |             |
| Cable      |             |          |             |             |
| Capsule    |             |          |             |             |
| Carpet     |             |          |             |             |
| Grid       |             |          |             |             |
| Hazelnut   |             |          |             |             |
| Leather    |             |          |             |             |
| Metal_nut  |             |          |             |             |
| Pill       |             |          |             |             |
| Screw      |             |          |             |             |
| Tile       |             |          |             |             |
| Toothbrush |             |          |             |             |
| Transistor |             |          |             |             |
| Wood       |             |          |             |             |
| Zipper     |             |          |             |             |
| Average    |             |          |             |             |

## [BTAD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

| Model   | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| ------- | ----------- | -------- | ----------- | ----------- |
| 01      |             |          |             |             |
| 02      |             |          |             |             |
| 03      |             |          |             |             |
| Average |             |          |             |             |

## [Visa Dataset](https://github.com/amazon-science/spot-diff)

| Model      | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| ---------- | ----------- | -------- | ----------- | ----------- |
| candle     |             |          |             |             |
| capsules   |             |          |             |             |
| cashew     |             |          |             |             |
| chewinggum |             |          |             |             |
| fryum      |             |          |             |             |
| macaroni1  |             |          |             |             |
| macaroni2  |             |          |             |             |
| pcb1       |             |          |             |             |
| pcb2       |             |          |             |             |
| pcb3       |             |          |             |             |
| pcb4       |             |          |             |             |
| pipe_fryum |             |          |             |             |
| Average    |             |          |             |             |

### Sample Results

![Sample Result 1](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/patchcore/results/0.png "Sample Result 1")

![Sample Result 2](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/patchcore/results/1.png "Sample Result 2")

![Sample Result 3](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/patchcore/results/2.png "Sample Result 3")
