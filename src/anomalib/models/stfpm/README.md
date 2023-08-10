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
| bottle     | 0.925       | 0.915    | 0.952       | 0.830       |
| cable      | 0.879       | 0.851    | 0.937       | 0.856       |
| capsule    | 0.897       | 0.922    | 0.974       | 0.898       |
| carpet     | 0.968       | 0.954    | 0.985       | 0.958       |
| grid       | 0.983       | 0.974    | 0.990       | 0.965       |
| hazelnut   | 1.000       | 1.000    | 0.988       | 0.969       |
| leather    | 1.000       | 1.000    | 0.996       | 0.988       |
| metal_nut  | 0.989       | 0.979    | 0.969       | 0.939       |
| pill       | 0.533       | 0.916    | 0.839       | 0.499       |
| screw      | 0.886       | 0.902    | 0.986       | 0.936       |
| tile       | 0.951       | 0.955    | 0.971       | 0.903       |
| toothbrush | 0.825       | 0.875    | 0.989       | 0.924       |
| transistor | 0.908       | 0.829    | 0.809       | 0.668       |
| wood       | 0.979       | 0.966    | 0.965       | 0.948       |
| zipper     | 0.832       | 0.920    | 0.979       | 0.940       |
| Average    | 0.904       | 0.931    | 0.955       | 0.881       |

## [BTAD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

| Category | image AUROC | image F1Score | pixel AUROC | pixel AUPRO |
| :------: | :---------: | :-----------: | :---------: | :---------: |
|    1     |    0.914    |     0.911     |    0.940    |    0.649    |
|    2     |    0.836    |     0.932     |    0.973    |    0.696    |
|    3     |    0.997    |     0.953     |    0.993    |    0.978    |
| Average  |    0.916    |     0.932     |    0.968    |    0.774    |

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
