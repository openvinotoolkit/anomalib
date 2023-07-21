# Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection

This is the implementation of the [STFPM](https://arxiv.org/pdf/2103.04257.pdf) paper.

Model Type: Segmentation

## Description

STFPM algorithm consists of a pre-trained teacher network and a student network with identical architecture. The student network learns the distribution of anomaly-free images by matching the features with the counterpart features in the teacher network. Multi-scale feature matching is used to enhance robustness. This hierarchical feature matching enables the student network to receive a mixture of multi-level knowledge from the feature pyramid thus allowing for anomaly detection of various sizes.

During inference, the feature pyramids of teacher and student networks are compared. Larger difference indicates a higher probability of anomaly occurrence.

## Architecture

![STFPM Architecture](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/stfpm/architecture.jpg "STFPM Architecture")

## Usage

`python tools/train.py --model stfpm`

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

![Sample Result 1](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/stfpm/results/0.png "Sample Result 1")

![Sample Result 2](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/stfpm/results/1.png "Sample Result 2")

![Sample Result 3](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/stfpm/results/2.png "Sample Result 3")
