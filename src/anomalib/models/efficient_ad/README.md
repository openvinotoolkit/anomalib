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

| Category   | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| ---------- | ----------- | -------- | ----------- | ----------- |
| bottle     | 1.000       | 1.000    | 0.984       | 0.944       |
| pill       | 0.986       | 0.975    | 0.985       | 0.956       |
| cable      | 0.942       | 0.919    | 0.982       | 0.916       |
| screw      | 0.973       | 0.952    | 0.985       | 0.959       |
| capsule    | 0.939       | 0.941    | 0.963       | 0.853       |
| tile       | 1.000       | 1.000    | 0.906       | 0.826       |
| toothbrush | 0.997       | 0.984    | 0.962       | 0.923       |
| carpet     | 0.990       | 0.978    | 0.965       | 0.929       |
| transistor | 0.947       | 0.900    | 0.946       | 0.819       |
| grid       | 0.999       | 0.991    | 0.937       | 0.889       |
| wood       | 0.968       | 0.952    | 0.870       | 0.778       |
| hazelnut   | 0.932       | 0.886    | 0.970       | 0.882       |
| zipper     | 0.971       | 0.975    | 0.960       | 0.930       |
| leather    | 0.999       | 0.995    | 0.976       | 0.975       |
| metalNut   | 0.979       | 0.978    | 0.978       | 0.917       |
| Average    | 0.975       | 0.962    | 0.958       | 0.900       |

## [BTAD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

| Category | image AUROC | image F1Score | pixel AUROC | pixel AUPRO |
| :------: | :---------: | :-----------: | :---------: | :---------: |
|    1     |    0.990    |     0.968     |    0.956    |    0.650    |
|    2     |    0.865    |     0.934     |    0.953    |    0.544    |
|    3     |    1.000    |     1.000     |    0.996    |    0.981    |
| Average  |    0.952    |     0.967     |    0.968    |    0.725    |

## [Visa Dataset](https://github.com/amazon-science/spot-diff)

| Model      | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| ---------- | ----------- | -------- | ----------- | ----------- |
| candle     | 0.639       | 0.674    | 0.854       | 0.574       |
| macaroni2  | 0.697       | 0.701    | 0.927       | 0.800       |
| capsules   | 0.302       | 0.769    | 0.877       | 0.595       |
| pcb1       | 0.909       | 0.818    | 0.984       | 0.866       |
| cashew     | 0.874       | 0.857    | 0.986       | 0.824       |
| pcb2       | 0.913       | 0.812    | 0.978       | 0.869       |
| chewinggum | 0.971       | 0.916    | 0.986       | 0.809       |
| pcb3       | 0.911       | 0.808    | 0.991       | 0.886       |
| fryum      | 0.830       | 0.847    | 0.974       | 0.821       |
| pcb4       | 0.980       | 0.922    | 0.960       | 0.763       |
| macaroni1  | 0.676       | 0.667    | 0.963       | 0.794       |
| pipe_fryum | 0.937       | 0.893    | 0.997       | 0.910       |
| Average    | 0.803       | 0.807    | 0.956       | 0.793       |
