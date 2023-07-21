# GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training

This is the implementation of the [GANomaly](https://arxiv.org/abs/1805.06725) paper.

Model Type: Classification

## Description

GANomaly uses the conditional GAN approach to train a Generator to produce images of the normal data. This Generator consists of an encoder-decoder-encoder architecture to generate the normal images. The distance between the latent vector $z$ between the first encoder-decoder and the output vector $\hat{z}$ is minimized during training.

The key idea here is that, during inference, when an anomalous image is passed through the first encoder the latent vector $z$ will not be able to capture the data correctly. This would leave to poor reconstruction $\hat{x}$ thus resulting in a very different $\hat{z}$. The difference between $z$ and $\hat{z}$ gives the anomaly score.

## Architecture

![GANomaly Architecture](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/ganomaly/architecture.jpg "GANomaly Architecture")

## Usage

`python tools/train.py --model ganomaly`

## Benchmark

All results gathered with seed `42`.

## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

| Model      | Image AUROC | Image F1 |
| ---------- | ----------- | -------- |
| Bottle     |             |          |
| Cable      |             |          |
| Capsule    |             |          |
| Carpet     |             |          |
| Grid       |             |          |
| Hazelnut   |             |          |
| Leather    |             |          |
| Metal_nut  |             |          |
| Pill       |             |          |
| Screw      |             |          |
| Tile       |             |          |
| Toothbrush |             |          |
| Transistor |             |          |
| Wood       |             |          |
| Zipper     |             |          |
| Average    |             |          |

## [BTAD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

| Model   | Image AUROC | Image F1 |
| ------- | ----------- | -------- |
| 01      |             |          |
| 02      |             |          |
| 03      |             |          |
| Average |             |          |

## [Visa Dataset](https://github.com/amazon-science/spot-diff)

| Model      | Image AUROC | Image F1 |
| ---------- | ----------- | -------- |
| candle     |             |          |
| capsules   |             |          |
| cashew     |             |          |
| chewinggum |             |          |
| fryum      |             |          |
| macaroni1  |             |          |
| macaroni2  |             |          |
| pcb1       |             |          |
| pcb2       |             |          |
| pcb3       |             |          |
| pcb4       |             |          |
| pipe_fryum |             |          |
| Average    |             |          |
