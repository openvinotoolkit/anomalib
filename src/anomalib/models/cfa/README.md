# CFA: Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cfa-coupled-hypersphere-based-feature/anomaly-detection-on-mvtec-ad)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad?p=cfa-coupled-hypersphere-based-feature)

This is the implementation of the [CFA](https://arxiv.org/abs/2206.04325) paper. The original implementation could be found [sungwool/cfa_for_anomaly_localization](https://github.com/sungwool/cfa_for_anomaly_localization).

Model Type: Segmentation

## Description

Coupled-hypersphere-based Feature Adaptation (CFA) localizes anomalies using features adapted to the target dataset. CFA consists of (1) a learnable patch descriptor that learns and embeds target-oriented features and (2) a scalable memory bank independent of the size of the target dataset. By applying a patch descriptor and memory bank to a pretrained CNN, CFA also employs transfer learning to increase the normal feature density so that abnormal features can be easily distinguished.

## Architecture

![Cfa Architecture](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/cfa/architecture.png "Cfa Architecture")

## Usage

`python tools/train.py --model cfa`

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

![Sample Result 1](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/cfa/results/0.png "Sample Result 1")

![Sample Result 2](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/cfa/results/1.png "Sample Result 2")

![Sample Result 3](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/cfa/results/2.png "Sample Result 3")

## Reference

[1] <https://github.com/sungwool/cfa_for_anomaly_localization>

## Citation

```tex
@article{lee2022cfa,
  title={CFA: Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization},
  author={Lee, Sungwook and Lee, Seunghyun and Song, Byung Cheol},
  journal={arXiv preprint arXiv:2206.04325},
  year={2022}
}
```
