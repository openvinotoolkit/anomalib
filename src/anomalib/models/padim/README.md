# PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization

This is the implementation of the [PaDiM](https://arxiv.org/pdf/2011.08785.pdf) paper.

Model Type: Segmentation

## Description

PaDiM is a patch based algorithm. It relies on a pre-trained CNN feature extractor. The image is broken into patches and embeddings are extracted from each patch using different layers of the feature extractors. The activation vectors from different layers are concatenated to get embedding vectors carrying information from different semantic levels and resolutions. This helps encode fine grained and global contexts. However, since the generated embedding vectors may carry redundant information, dimensions are reduced using random selection. A multivariate gaussian distribution is generated for each patch embedding across the entire training batch. Thus, for each patch of the set of training images, we have a different multivariate gaussian distribution. These gaussian distributions are represented as a matrix of gaussian parameters.

During inference, Mahalanobis distance is used to score each patch position of the test image. It uses the inverse of the covariance matrix calculated for the patch during training. The matrix of Mahalanobis distances forms the anomaly map with higher scores indicating anomalous regions.

## Architecture

![PaDiM Architecture](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/padim/architecture.jpg "PaDiM Architecture")

## Usage

`python tools/train.py --model padim`

## Benchmark

All results gathered with seed `42`.

## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

| Model      | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| ---------- | ----------- | -------- | ----------- | ----------- |
| Bottle     | 0.9937      | 0.9764   | 0.9830      | 0.9511      |
| Cable      | 0.8433      | 0.8585   | 0.9645      | 0.9036      |
| Capsule    | 0.9015      | 0.9604   | 0.9843      | 0.9170      |
| Carpet     | 0.9454      | 0.9302   | 0.9835      | 0.9487      |
| Grid       | 0.8571      | 0.8926   | 0.9177      | 0.8092      |
| Hazelnut   | 0.7507      | 0.8364   | 0.9779      | 0.9414      |
| Leather    | 0.9823      | 0.9838   | 0.9937      | 0.9826      |
| Metal_nut  | 0.9614      | 0.9738   | 0.9696      | 0.9144      |
| Pill       | 0.8628      | 0.9324   | 0.9570      | 0.9375      |
| Screw      | 0.7588      | 0.8788   | 0.9782      | 0.9227      |
| Tile       | 0.9502      | 0.9341   | 0.9339      | 0.8171      |
| Toothbrush | 0.8889      | 0.9231   | 0.9882      | 0.9327      |
| Transistor | 0.9200      | 0.7957   | 0.9679      | 0.9152      |
| Wood       | 0.9763      | 0.9516   | 0.9475      | 0.9234      |
| Zipper     | 0.7797      | 0.9154   | 0.9789      | 0.9281      |
| Average    | 0.8915      | 0.9162   | 0.9684      | 0.9163      |

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

![Sample Result 1](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/padim/results/0.png "Sample Result 1")

![Sample Result 2](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/padim/results/1.png "Sample Result 2")

![Sample Result 3](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/padim/results/2.png "Sample Result 3")
