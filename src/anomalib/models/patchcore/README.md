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
| Bottle     | 1.0000      | 1.0000   | 0.9796      | 0.9207      |
| Cable      | 0.9871      | 0.9783   | 0.9795      | 0.9036      |
| Capsule    | 0.9761      | 0.9774   | 0.9873      | 0.9187      |
| Carpet     | 0.9779      | 0.9714   | 0.9866      | 0.9279      |
| Grid       | 0.9724      | 0.9649   | 0.9790      | 0.8986      |
| Hazelnut   | 1.0000      | 1.0000   | 0.9841      | 0.9429      |
| Leather    | 1.0000      | 1.0000   | 0.9884      | 0.9630      |
| Metal_nut  | 0.9971      | 0.9892   | 0.9844      | 0.9154      |
| Pill       | 0.9370      | 0.9534   | 0.9736      | 0.9365      |
| Screw      | 0.9762      | 0.9661   | 0.9908      | 0.9423      |
| Tile       | 0.9874      | 0.9880   | 0.9471      | 0.7902      |
| Toothbrush | 1.0000      | 1.0000   | 0.9866      | 0.8594      |
| Transistor | 1.0000      | 1.0000   | 0.9648      | 0.9378      |
| Wood       | 0.9930      | 0.9831   | 0.9310      | 0.8441      |
| Zipper     | 0.9937      | 0.9874   | 0.9799      | 0.9216      |
| Average    | 0.9865      | 0.9839   | 0.9762      | 0.9082      |

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
