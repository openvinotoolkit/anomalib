# Real-Time Unsupervised Anomaly Detection via Conditional Normalizing Flows

This is the implementation of the [CFLOW-AD](https://arxiv.org/pdf/2107.12571v1.pdf) paper. This code is modified form of the [official repository](https://github.com/gudovskiy/cflow-ad).

Model Type: Segmentation

## Description

CFLOW model is based on a conditional normalizing flow framework adopted for anomaly detection with localization. It consists of a discriminatively pretrained encoder followed by a multi-scale generative decoders. The encoder extracts features with multi-scale pyramid pooling to capture both global and local semantic information with the growing from top to bottom receptive fields. Pooled features are processed by a set of decoders to explicitly estimate likelihood of the encoded features. The estimated multi-scale likelyhoods are upsampled to input size and added up to produce the anomaly map.

## Architecture

![CFlow Architecture](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/cflow/architecture.jpg "CFlow Architecture")

## Usage

`python tools/train.py --model cflow`

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

![Sample Result 1](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/cflow/results/0.png "Sample Result 1")

![Sample Result 2](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/cflow/results/1.png "Sample Result 2")

![Sample Result 3](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/cflow/results/2.png "Sample Result 3")
