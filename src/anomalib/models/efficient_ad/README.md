# EfficientAd

This is the implementation of the [EfficientAd](https://arxiv.org/pdf/2303.14535.pdf) paper. It is based on https://github.com/rximg/EfficientAd and https://github.com/nelson1425/EfficientAd/

Model Type: Segmentation

## Description

Fast anomaly segmentation algorithm that consists of a distilled pre-trained teacher model, a student model and an autoencoder. It detects local anomalies via the teacher-student discrepany and global anomalies via the student-autoencoder discrepancy.

### Feature Extraction

Features are extracted from a pre-trained teacher model and used to train a student model and an autoencoder model. To hinder the student from imitating the teacher on anomalies, Imagenet images are used in the loss function.

### Anomaly Detection

Anomalies are detected as the difference in output feature maps between the teacher model, the student model and the autoencoder model.

## Usage

`python tools/train.py --model efficient_ad`

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
