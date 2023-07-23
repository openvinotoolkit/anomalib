# FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows

This is the implementation of the [FastFlow](https://arxiv.org/abs/2111.07677) paper. This code is developed by utilizing the torch model implemented in [https://github.com/gathierry/FastFlow](https://github.com/gathierry/FastFlow).

Model Type: Segmentation

## Description

FastFlow is a two-dimensional normalizing flow-based probability distribution estimator. It can be used as a plug-in module with any deep feature extractor, such as ResNet and vision transformer, for unsupervised anomaly detection and localisation. In the training phase, FastFlow learns to transform the input visual feature into a tractable distribution, and in the inference phase, it assesses the likelihood of identifying anomalies.

## Architecture

![FastFlow Architecture](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/fastflow/architecture.jpg "FastFlow Architecture")

## Usage

`python tools/train.py --model fastflow`

## Benchmark

All results gathered with seed `42`.

## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

| Model      | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| ---------- | ----------- | -------- | ----------- | ----------- |
| bottle     | 0.9690      | 0.9545   | 0.9646      | 0.8473      |
| pill       | 0.9373      | 0.9622   | 0.9801      | 0.9165      |
| cable      | 0.9809      | 0.9534   | 0.9715      | 0.8687      |
| screw      | 0.8953      | 0.9077   | 0.9782      | 0.9113      |
| capsule    | 0.9561      | 0.9545   | 0.9838      | 0.9021      |
| tile       | 0.9643      | 0.9425   | 0.9085      | 0.7697      |
| carpet     | 0.9980      | 0.9888   | 0.9903      | 0.9667      |
| grid       | 0.9916      | 0.9825   | 0.9790      | 0.9205      |
| toothbrush | 0.8611      | 0.8824   | 0.9823      | 0.7928      |
| hazelnut   | 0.9854      | 0.9718   | 0.9879      | 0.9420      |
| transistor | 0.9937      | 0.9512   | 0.9538      | 0.8594      |
| leather    | 1.0000      | 1.0000   | 0.9909      | 0.9586      |
| wood       | 0.9649      | 0.9440   | 0.9530      | 0.9216      |
| metal_nut  | 0.9990      | 0.9946   | 0.9839      | 0.8966      |
| zipper     | 0.9827      | 0.9636   | 0.9769      | 0.9182      |
| Average    | 0.9653      | 0.9569   | 0.9723      | 0.8928      |

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

![Sample Result 1](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/fastflow/results/0.png "Sample Result 1")

![Sample Result 2](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/fastflow/results/1.png "Sample Result 2")

![Sample Result 3](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/fastflow/results/2.png "Sample Result 3")
