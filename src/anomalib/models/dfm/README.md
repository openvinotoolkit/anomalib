# Probabilistic Modeling of Deep Features for Out-of-Distribution and Adversarial Detection

This is the implementation of [DFM](https://arxiv.org/pdf/1909.11786.pdf) paper.

Model Type: Classification

## Description

Fast anomaly classification algorithm that consists of a deep feature extraction stage followed by anomaly classification stage consisting of PCA and class-conditional Gaussian Density Estimation.

### Feature Extraction

Features are extracted by feeding the images through a ResNet18 backbone, which was pre-trained on ImageNet. The output of the penultimate layer (average pooling layer) of the network is used to obtain a semantic feature vector with a fixed length of 2048.

### Anomaly Detection

In the anomaly classification stage, class-conditional PCA transformations and Gaussian Density models are learned. Two types of scores are calculated (i) Feature-reconstruction scores (norm of the difference between the high-dimensional pre-image of a reduced dimension feature and the original high-dimensional feature), and (ii) Negative log-likelihood under the learnt density models. Anomaly map generation is supported only with the feature-reconstruction based scores. Image level anomaly detection is supported by both score types.

## Usage

`python tools/train.py --model dfm`

## Benchmark

All results gathered with seed `42`.

## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

| Category   | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| ---------- | ----------- | -------- | ----------- | ----------- |
| bottle     | 0.998       | 0.992    | 0.972       | 0.878       |
| pill       | 0.954       | 0.946    | 0.970       | 0.885       |
| cable      | 0.948       | 0.912    | 0.974       | 0.864       |
| screw      | 0.863       | 0.890    | 0.973       | 0.868       |
| capsule    | 0.946       | 0.960    | 0.981       | 0.855       |
| tile       | 0.986       | 0.982    | 0.943       | 0.852       |
| carpet     | 0.943       | 0.920    | 0.984       | 0.907       |
| grid       | 0.790       | 0.868    | 0.926       | 0.796       |
| toothbrush | 0.967       | 0.966    | 0.983       | 0.782       |
| hazelnut   | 0.980       | 0.964    | 0.982       | 0.933       |
| transistor | 0.963       | 0.889    | 0.985       | 0.925       |
| leather    | 1.000       | 1.000    | 0.982       | 0.960       |
| metal_nut  | 0.922       | 0.929    | 0.973       | 0.861       |
| wood       | 0.994       | 0.984    | 0.905       | 0.836       |
| zipper     | 0.964       | 0.971    | 0.958       | 0.850       |
| Average    | 0.948       | 0.945    | 0.966       | 0.870       |

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
| candle     | 0.947       | 0.892    | 0.978       | 0.912       |
| capsules   | 0.671       | 0.774    | 0.944       | 0.456       |
| cashew     | 0.935       | 0.911    | 0.988       | 0.710       |
| chewinggum | 0.986       | 0.959    | 0.978       | 0.797       |
| fryum      | 0.938       | 0.923    | 0.971       | 0.744       |
| macaroni1  | 0.769       | 0.754    | 0.949       | 0.869       |
| macaroni2  | 0.730       | 0.747    | 0.920       | 0.790       |
| pcb1       | 0.934       | 0.861    | 0.991       | 0.865       |
| pcb2       | 0.841       | 0.784    | 0.963       | 0.784       |
| pcb3       | 0.830       | 0.763    | 0.961       | 0.684       |
| pcb4       | 0.972       | 0.935    | 0.973       | 0.852       |
| pipe_fryum | 0.812       | 0.840    | 0.992       | 0.885       |
| Average    | 0.864       | 0.845    | 0.967       | 0.779       |
