# Anomaly Detection via Reverse Distillation from One-Class Embedding

This is the implementation of the [Reverse Distillation](https://arxiv.org/pdf/2201.10703v2.pdf) paper.

Model Type: Segmentation

## Description

Reverse Distillation model consists of three networks. The first is a pre-trained feature extractor. The next two are the one-class embedding and the decoder networks.


## Architecture

![Anomaly Detection via Reverse Distillation from One-Class Embedding Architecture](../../../docs/source/images/reversedistillation/architecture.png "Reverse Distillation Architecture")

## Usage

`python tools/train.py --model reversedistillation`

## Benchmark

All results gathered with seed `42`.

## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Image-Level AUC

### Pixel-Level AUC

### Image F1 Score

### Sample Results
