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
| bottle     | 0.9984      | 0.9920   | 0.9717      | 0.8778      |
| pill       | 0.9536      | 0.9455   | 0.9703      | 0.8849      |
| cable      | 0.9479      | 0.9119   | 0.9739      | 0.8643      |
| screw      | 0.8629      | 0.8898   | 0.9731      | 0.8676      |
| capsule    | 0.9462      | 0.9600   | 0.9807      | 0.8552      |
| tile       | 0.9856      | 0.9818   | 0.9425      | 0.8522      |
| carpet     | 0.9426      | 0.9195   | 0.9836      | 0.9072      |
| grid       | 0.7895      | 0.8682   | 0.9264      | 0.7961      |
| toothbrush | 0.9667      | 0.9655   | 0.9834      | 0.7821      |
| hazelnut   | 0.9804      | 0.9640   | 0.9823      | 0.9332      |
| transistor | 0.9625      | 0.8889   | 0.9854      | 0.9249      |
| leather    | 1.0000      | 1.0000   | 0.9818      | 0.9599      |
| metal_nut  | 0.9223      | 0.9286   | 0.9732      | 0.8610      |
| wood       | 0.9939      | 0.9836   | 0.9048      | 0.8362      |
| zipper     | 0.9638      | 0.9710   | 0.9580      | 0.8502      |
| Average    | 0.9477      | 0.9447   | 0.9661      | 0.8702      |

## [BTAD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

| Model   | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| ------- | ----------- | -------- | ----------- | ----------- |
| 01      |             |          |             |             |
| 02      |             |          |             |             |
| 03      |             |          |             |             |
| Average |             |          |             |             |

## [Visa Dataset](https://github.com/amazon-science/spot-diff)

| Model      | Image AUROC | Image F1   | Pixel AUROC | Pixel AUPRO |
| ---------- | ----------- | ---------- | ----------- | ----------- |
| candle     | 0.94679999  | 0.8921569  | 0.97814381  | 0.91176367  |
| capsules   | 0.67116666  | 0.77366251 | 0.94374216  | 0.45603833  |
| cashew     | 0.93540001  | 0.91079819 | 0.98766309  | 0.71012241  |
| chewinggum | 0.98640001  | 0.95918369 | 0.97767627  | 0.79671788  |
| fryum      | 0.93779999  | 0.92307687 | 0.97122395  | 0.74374688  |
| macaroni1  | 0.76919997  | 0.75396824 | 0.9490428   | 0.86897069  |
| macaroni2  | 0.7299      | 0.74698794 | 0.91983593  | 0.79018855  |
| pcb1       | 0.93400002  | 0.86124402 | 0.99148774  | 0.86543226  |
| pcb2       | 0.84109998  | 0.78378379 | 0.96313369  | 0.78399605  |
| pcb3       | 0.82980204  | 0.76305223 | 0.96070528  | 0.68428123  |
| pcb4       | 0.97227728  | 0.93532342 | 0.97289878  | 0.85156047  |
| pipe_fryum | 0.81239998  | 0.83962262 | 0.99198556  | 0.88500786  |
| Average    | 0.86385383  | 0.84523837 | 0.96729492  | 0.77898552  |
