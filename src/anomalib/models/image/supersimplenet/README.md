# SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection

This is an implementation of the [SuperSimpleNet](https://arxiv.org/pdf/2408.03143) paper, based on the [official code](https://github.com/blaz-r/SuperSimpleNet).

Model Type: Segmentation

## Description

**SuperSimpleNet** is a simple yet strong discriminative defect / anomaly detection model evolved from the SimpleNet architecture. It consists of four components:
feature extractor with upscaling, feature adaptor, synthetic feature-level anomaly generation module, and
segmentation-detection module. 

A ResNet-like feature extractor first extracts features, which are then upscaled and
average-pooled to capture neighboring context. Features are further refined for anomaly detection task in the adaptor module.
During training, synthetic anomalies are generated at the feature level by adding Gaussian noise to regions defined by the
binary Perlin noise mask. The perturbed features are then fed into the segmentation-detection
module, which produces the anomaly map and the anomaly score. During inference, anomaly generation is skipped, and the model
directly predicts the anomaly map and score. The predicted anomaly map is upscaled to match the input image size
and refined with a Gaussian filter.

This implementation supports both unsupervised and supervised setting, but Anomalib currently supports only unsupervised learning.

## Architecture

![SuperSimpleNet architecture](/docs/source/images/supersimplenet/architecture.png "SuperSimpleNet architecture")

## Usage

`anomalib train --model SuperSimpleNet --data MVTec --data.category <category>`

> It is recommended to train the model for 300 epochs with batch size of 32 to achieve stable training with random anomaly generation. Training with lower parameter values will still work, but might not yield the optimal results.
>
> For supervised learning, refer to the [official code](https://github.com/blaz-r/SuperSimpleNet).

## MVTec AD results

The following results were obtained using this Anomalib implementation trained for 300 epochs with seed 42, default params, and batch size 32. 
|             | **Image AUROC** | **Pixel AUPRO** |
| ----------- | :-------------: | :-------------: |
| Bottle      |      1.000      |      0.914      |
| Cable       |      0.981      |      0.895      |
| Capsule     |      0.990      |      0.926      |
| Carpet      |      0.987      |      0.936      |
| Grid        |      0.998      |      0.935      |
| Hazelnut    |      0.999      |      0.946      |
| Leather     |      1.000      |      0.972      |
| Metal_nut   |      0.996      |      0.923      |
| Pill        |      0.960      |      0.942      |
| Screw       |      0.903      |      0.952      |
| Tile        |      0.989      |      0.817      |
| Toothbrush  |      0.917      |      0.861      |
| Transistor  |      1.000      |      0.909      |
| Wood        |      0.996      |      0.868      |
| Zipper      |      0.996      |      0.944      |
| **Average** |      0.981      |      0.916      |

For other results on VisA, SensumSODF, and KSDD2, refer to the [paper](https://arxiv.org/pdf/2408.03143).
