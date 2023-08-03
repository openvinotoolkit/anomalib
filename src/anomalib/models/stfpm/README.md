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
| bottle     | 0.9246      | 0.9153   | 0.9519      | 0.8297      |
| cable      | 0.8787      | 0.8513   | 0.9373      | 0.8555      |
| capsule    | 0.8971      | 0.9217   | 0.9743      | 0.8976      |
| carpet     | 0.9679      | 0.9535   | 0.9850      | 0.9581      |
| grid       | 0.9833      | 0.9739   | 0.9896      | 0.9646      |
| hazelnut   | 1.0000      | 1.0000   | 0.9882      | 0.9686      |
| leather    | 1.0000      | 1.0000   | 0.9955      | 0.9884      |
| metal_nut  | 0.9888      | 0.9787   | 0.9689      | 0.9389      |
| pill       | 0.5333      | 0.9156   | 0.8394      | 0.4992      |
| screw      | 0.8862      | 0.9024   | 0.9860      | 0.9360      |
| tile       | 0.9513      | 0.9545   | 0.9705      | 0.9027      |
| toothbrush | 0.8250      | 0.8750   | 0.9893      | 0.9242      |
| transistor | 0.9079      | 0.8293   | 0.8093      | 0.6677      |
| wood       | 0.9789      | 0.9661   | 0.9645      | 0.9479      |
| zipper     | 0.8319      | 0.9200   | 0.9794      | 0.9399      |
| Average    | 0.9037      | 0.9305   | 0.9553      | 0.8813      |

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
| candle     | 0.905       | 0.853    | 0.973       | 0.925       |
| capsules   | 0.820       | 0.822    | 0.964       | 0.909       |
| cashew     | 0.883       | 0.865    | 0.864       | 0.769       |
| chewinggum | 0.957       | 0.954    | 0.973       | 0.717       |
| fryum      | 0.835       | 0.847    | 0.923       | 0.803       |
| macaroni1  | 0.889       | 0.820    | 0.995       | 0.947       |
| macaroni2  | 0.854       | 0.789    | 0.991       | 0.945       |
| pcb1       | 0.923       | 0.893    | 0.992       | 0.926       |
| pcb2       | 0.920       | 0.880    | 0.983       | 0.850       |
| pcb3       | 0.896       | 0.818    | 0.981       | 0.899       |
| pcb4       | 0.972       | 0.951    | 0.947       | 0.720       |
| pipe_fryum | 0.954       | 0.938    | 0.986       | 0.920       |
| Average    | 0.901       | 0.869    | 0.964       | 0.861       |

### Sample Results

![Sample Result 1](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/stfpm/results/0.png "Sample Result 1")

![Sample Result 2](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/stfpm/results/1.png "Sample Result 2")

![Sample Result 3](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/stfpm/results/2.png "Sample Result 3")
