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
| 01      | 0.9951      | 0.9796   | 0.9654      | 0.7576      |
| 02      | 0.8613      | 0.9302   | 0.9612      | 0.6145      |
| 03      | 0.9765      | 0.7711   | 0.9953      | 0.9832      |
| Average | 0.9443      | 0.8936   | 0.9740      | 0.7851      |

## [Visa Dataset](https://github.com/amazon-science/spot-diff)

| Model      | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| ---------- | ----------- | -------- | ----------- | ----------- |
| candle     | 0.8619      | 0.8387   | 0.9768      | 0.9277      |
| capsules   | 0.6092      | 0.7692   | 0.9259      | 0.5636      |
| cashew     | 0.8848      | 0.8547   | 0.9783      | 0.8338      |
| chewinggum | 0.9814      | 0.9749   | 0.9885      | 0.8417      |
| fryum      | 0.8582      | 0.8792   | 0.9582      | 0.7623      |
| macaroni1  | 0.7805      | 0.7611   | 0.9802      | 0.8880      |
| macaroni2  | 0.7191      | 0.7212   | 0.9602      | 0.7533      |
| pcb1       | 0.8724      | 0.8267   | 0.9872      | 0.8778      |
| pcb2       | 0.7882      | 0.7444   | 0.9804      | 0.8370      |
| pcb3       | 0.7154      | 0.7016   | 0.9797      | 0.7914      |
| pcb4       | 0.9683      | 0.9458   | 0.9680      | 0.7995      |
| pipe_fryum | 0.9140      | 0.8976   | 0.9916      | 0.8787      |
| Average    | 0.8295      | 0.8263   | 0.9729      | 0.8129      |

### Sample Results

![Sample Result 1](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/padim/results/0.png "Sample Result 1")

![Sample Result 2](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/padim/results/1.png "Sample Result 2")

![Sample Result 3](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/padim/results/2.png "Sample Result 3")
