# U-Flow: A U-shaped Normalizing Flow for Anomaly Detection with Unsupervised Threshold

[//]: # "This is the implementation of the [U-Flow](https://arxiv.org/abs/2211.12353) paper, based on the [original code](https://www.github.com/mtailanian/uflow)"

This is the implementation of the [U-Flow](https://www.researchsquare.com/article/rs-3367286/latest) paper, based on the [original code](https://www.github.com/mtailanian/uflow)

![U-Flow Architecture](/docs/source/images/uflow/diagram.png "U-Flow Architecture")

## Abstract

_In this work we propose a one-class self-supervised method for anomaly segmentation in images, that benefits both from a modern machine learning approach and a more classic statistical detection theory.
The method consists of three phases. First, features are extracted using a multi-scale image Transformer architecture. Then, these features are fed into a U-shaped Normalizing Flow that lays the theoretical foundations for the last phase, which computes a pixel-level anomaly map and performs a segmentation based on the a contrario framework.
This multiple-hypothesis testing strategy permits the derivation of robust automatic detection thresholds, which are crucial in real-world applications where an operational point is needed.
The segmentation results are evaluated using the Intersection over Union (IoU) metric, and for assessing the generated anomaly maps we report the area under the Receiver Operating Characteristic curve (AUROC), and the area under the per-region-overlap curve (AUPRO).
Extensive experimentation in various datasets shows that the proposed approach produces state-of-the-art results for all metrics and all datasets, ranking first in most MvTec-AD categories, with a mean pixel-level AUROC of 98.74%._

![Teaser image](/docs/source/images/uflow/teaser.jpg)

## Localization results

### Pixel AUROC over MVTec-AD Dataset

![Pixel-AUROC results](/docs/source/images/uflow/pixel-auroc.png "Pixel-AUROC results")

### Pixel AUPRO over MVTec-AD Dataset

![Pixel-AUPRO results](/docs/source/images/uflow/pixel-aupro.png "Pixel-AUPRO results")

## Segmentation results (IoU) with threshold log(NFA)=0

This paper also proposes a method to automatically compute the threshold using the a contrario framework. All results below are obtained with the threshold log(NFA)=0.
In the default code here, for the sake of comparison with all the other methods of the library, the segmentation is done computing the threshold over the anomaly map at train time.
Nevertheless, the code for computing the segmentation mask with the NFA criterion is included in the `src/anomalib/models/uflow/anomaly_map.py`.

![IoU results](/docs/source/images/uflow/iou.png "IoU results")

## Results over other datasets

![Results over other datasets](/docs/source/images/uflow/more-results.png "Results over other datasets")

## Benchmarking

Note that the proposed method uses the MCait Feature Extractor, which has an input size of 448x448. In the benchmarking, a size of 256x256 is used for all methods, and therefore the results may differ from those reported. In order to exactly reproduce all results, the reader can refer to the original code (see [here](https://www.github.com/mtailanian/uflow), where the configs used and even the trained checkpoints can be downloaded from [this release](https://github.com/mtailanian/uflow/releases/tag/trained-models-for-all-mvtec-categories).

## Reproducing paper's results

Using the default parameters of the config file (`src/anomalib/models/uflow/config.yaml`), the results obtained are very close to the ones reported in the paper:

bottle: 97.98, cable: 98.17, capsule: 98.95, carpet: 99.45, grid: 98.19, hazelnut: 99.01, leather: 99.41, metal_nut: 98.19, pill: 99.15, screw: 99.25, tile: 96.93, toothbrush: 98.97, transistor: 96.70, wood: 96.87, zipper: 97.92

In order to obtain the same exact results, although the architecture parameters stays always the same, the following values for the learning rate and batch size should be used (please refer to the [original code](https://www.github.com/mtailanian/uflow) for more details, where the used configs are available in the source code ([here](https://github.com/mtailanian/uflow/tree/main/configs)), and trained checkpoints are available in [this release](https://github.com/mtailanian/uflow/releases/tag/trained-models-for-all-mvtec-categories)):

## Usage

`anomalib train --model Uflow --data MVTec --data.category <category>`

## Download data

### MVTec

https://www.mvtec.com/company/research/datasets/mvtec-ad

### Bean Tech

https://paperswithcode.com/dataset/btad

### LGG MRI

https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

### ShanghaiTech Campus

https://svip-lab.github.io/dataset/campus_dataset.html

## [Optional] Download pre-trained models

Pre-trained models can be found in [this release](https://github.com/mtailanian/uflow/tree/main/configs), or can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1W1rE0mu4Lv3uWHA5GZigmvVNlBVHqTv_?usp=sharing)

For an easier way of downloading them, please refer to the `README.md` from the [original code](https://www.github.com/mtailanian/uflow)

For reproducing the exact results from the paper, different learning rates and batch sizes are to be used for each category. You can find the exact values in the `configs` folder, following the [previous link](https://drive.google.com/drive/folders/1W1rE0mu4Lv3uWHA5GZigmvVNlBVHqTv_?usp=sharing).

## A note on sizes at different points

Input

```text
- Scale 1: [3, 448, 448]
- Scale 2: [3, 224, 224]
```

MS-Cait outputs

```text
- Scale 1: [768, 28, 28]
- Scale 2: [384, 14, 14]
```

Normalizing Flow outputs

```text
- Scale 1: [816, 28, 28] --> 816 = 768 + 384 / 2 / 4
- Scale 2: [192, 14, 14] --> 192 = 384 / 2
```

`/ 2` corresponds to the split, and `/ 4` to the invertible upsample.

## Example results

### Anomalies

#### MVTec

![MVTec results - anomalies](/docs/source/images/uflow/results-mvtec-anomalies.jpg "MVTec results - anomalies")

#### BeanTech, LGG MRI, STC

![BeanTech, LGG MRI, STC results - anomalies](/docs/source/images/uflow/results-others-anomalies.jpg "BeanTech, LGG MRI, STC results - anomalies")

### Normal images

#### MVTec

![MVTec results - normal](/docs/source/images/uflow/results-mvtec-good.jpg "MVTec results - normal")

#### BeanTech, LGG MRI, STC

![BeanTech, LGG MRI, STC results - normal](/docs/source/images/uflow/results-others-good.jpg "BeanTech, LGG MRI, STC results - normal")
