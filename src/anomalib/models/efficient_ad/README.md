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

|  Category  | image AUROC | image F1Score | pixel AUROC | pixel AUPRO |
| :--------: | :---------: | :-----------: | :---------: | :---------: |
|   Bottle   |    1.000    |     1.000     |    0.984    |    0.944    |
|   Cable    |    0.942    |     0.919     |    0.982    |    0.916    |
|  Capsule   |    0.939    |     0.941     |    0.963    |    0.853    |
|   Carpet   |    0.990    |     0.978     |    0.965    |    0.929    |
|    Grid    |    0.999    |     0.991     |    0.937    |    0.889    |
|  Hazelnut  |    0.932    |     0.886     |    0.970    |    0.882    |
|  Leather   |    0.999    |     0.995     |    0.976    |    0.975    |
| Metal_Nut  |    0.979    |     0.978     |    0.978    |    0.917    |
|    Pill    |    0.986    |     0.975     |    0.985    |    0.956    |
|   Screw    |    0.973    |     0.952     |    0.985    |    0.959    |
|    Tile    |    1.000    |     1.000     |    0.906    |    0.826    |
| Toothbrush |    0.997    |     0.984     |    0.962    |    0.923    |
| Transistor |    0.947    |     0.900     |    0.946    |    0.819    |
|    Wood    |    0.968    |     0.952     |    0.870    |    0.778    |
|   Zipper   |    0.971    |     0.975     |    0.960    |    0.930    |
|  Average   |    0.975    |     0.962     |    0.958    |    0.900    |

## [BTAD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

| Category | image AUROC | image F1Score | pixel AUROC | pixel AUPRO |
| :------: | :---------: | :-----------: | :---------: | :---------: |
|    1     |    0.941    |     0.918     |    0.668    |    0.323    |
|    3     |    0.996    |     0.914     |    0.964    |    0.898    |
|    2     |    0.776    |     0.930     |    0.845    |    0.394    |
| Average  |    0.904    |     0.921     |    0.825    |    0.539    |

## [Visa Dataset](https://github.com/amazon-science/spot-diff)

|  Category  | image AUROC | image F1Score | pixel AUROC | pixel AUPRO |
| :--------: | :---------: | :-----------: | :---------: | :---------: |
|   candle   |    0.639    |     0.674     |    0.854    |    0.574    |
|  capsules  |    0.302    |     0.769     |    0.877    |    0.595    |
|   cashew   |    0.874    |     0.857     |    0.986    |    0.824    |
| chewinggum |    0.971    |     0.916     |    0.986    |    0.809    |
|   fryum    |    0.830    |     0.847     |    0.974    |    0.821    |
| macaroni1  |    0.676    |     0.667     |    0.963    |    0.794    |
| macaroni2  |    0.697    |     0.701     |    0.927    |    0.800    |
|    pcb1    |    0.909    |     0.818     |    0.984    |    0.866    |
|    pcb2    |    0.913    |     0.812     |    0.978    |    0.869    |
|    pcb3    |    0.911    |     0.808     |    0.991    |    0.886    |
|    pcb4    |    0.980    |     0.922     |    0.960    |    0.763    |
| pipe_fryum |    0.937    |     0.893     |    0.997    |    0.910    |
|  Average   |    0.803    |     0.807     |    0.956    |    0.793    |
