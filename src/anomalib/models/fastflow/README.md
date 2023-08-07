# FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows

This is the implementation of the [FastFlow](https://arxiv.org/abs/2111.07677) paper. This code is developed by utilizing the torch model implemented in [https://github.com/gathierry/FastFlow](https://github.com/gathierry/FastFlow).

Model Type: Segmentation

## Description

FastFlow is a two-dimensional normalizing flow-based probability distribution estimator. It can be used as a plug-in module with any deep feature extractor, such as ResNet and vision transformer, for unsupervised anomaly detection and localisation. In the training phase, FastFlow learns to transform the input visual feature into a tractable distribution, and in the inference phase, it assesses the likelihood of identifying anomalies.

## Architecture

![FastFlow Architecture](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/fastflow/architecture.jpg "FastFlow Architecture")

## Usage

`python tools/train.py --model fastflow`

## Benchmark

All results gathered with seed `42`.

## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

| Category   | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| ---------- | ----------- | -------- | ----------- | ----------- |
| bottle     | 0.969       | 0.955    | 0.965       | 0.847       |
| pill       | 0.937       | 0.962    | 0.980       | 0.917       |
| cable      | 0.981       | 0.953    | 0.972       | 0.869       |
| screw      | 0.895       | 0.908    | 0.978       | 0.911       |
| capsule    | 0.956       | 0.955    | 0.984       | 0.902       |
| tile       | 0.964       | 0.943    | 0.909       | 0.770       |
| carpet     | 0.998       | 0.989    | 0.990       | 0.967       |
| grid       | 0.992       | 0.983    | 0.979       | 0.921       |
| toothbrush | 0.861       | 0.882    | 0.982       | 0.793       |
| hazelnut   | 0.985       | 0.972    | 0.988       | 0.942       |
| transistor | 0.994       | 0.951    | 0.954       | 0.859       |
| leather    | 1.000       | 1.000    | 0.991       | 0.959       |
| wood       | 0.965       | 0.944    | 0.953       | 0.922       |
| metal_nut  | 0.999       | 0.995    | 0.984       | 0.897       |
| zipper     | 0.983       | 0.964    | 0.977       | 0.918       |
| Average    | 0.965       | 0.957    | 0.972       | 0.893       |

## [BTAD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

| Category | image AUROC | image F1Score | pixel AUROC | pixel AUPRO |
| :------: | :---------: | :-----------: | :---------: | :---------: |
|    1     |    0.991    |     0.970     |    0.929    |    0.538    |
|    2     |    0.791    |     0.930     |    0.932    |    0.488    |
|    3     |    0.980    |     0.814     |    0.978    |    0.888    |
| Average  |    0.921    |     0.905     |    0.946    |    0.638    |

## [Visa Dataset](https://github.com/amazon-science/spot-diff)

| Model      | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| ---------- | ----------- | -------- | ----------- | ----------- |
| candle     | 0.923       | 0.854    | 0.959       | 0.922       |
| capsules   | 0.807       | 0.829    | 0.991       | 0.869       |
| cashew     | 0.853       | 0.850    | 0.993       | 0.884       |
| chewinggum | 0.989       | 0.965    | 0.986       | 0.845       |
| fryum      | 0.909       | 0.894    | 0.851       | 0.555       |
| macaroni1  | 0.951       | 0.897    | 0.990       | 0.934       |
| macaroni2  | 0.781       | 0.716    | 0.917       | 0.756       |
| pcb1       | 0.953       | 0.877    | 0.996       | 0.925       |
| pcb2       | 0.925       | 0.859    | 0.987       | 0.873       |
| pcb3       | 0.941       | 0.862    | 0.980       | 0.718       |
| pcb4       | 0.951       | 0.876    | 0.978       | 0.842       |
| pipe_fryum | 0.912       | 0.913    | 0.984       | 0.827       |
| Average    | 0.908       | 0.866    | 0.968       | 0.829       |

### Sample Results

![Sample Result 1](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/fastflow/results/0.png "Sample Result 1")

![Sample Result 2](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/fastflow/results/1.png "Sample Result 2")

![Sample Result 3](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/fastflow/results/2.png "Sample Result 3")
