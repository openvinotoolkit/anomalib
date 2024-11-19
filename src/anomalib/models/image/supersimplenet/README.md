# SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection

This is the implementation of the [SuperSimpleNet](https://arxiv.org/pdf/2408.03143) paper.

Model Type: Segmentation

## Description

SuperSimpleNet is a discriminative defect / anomaly detection model evolved from the SimpleNet architecture. It consists of four components:
feature extractor with upscaling, feature adaptor, synthetic feature-level anomaly generation module, and
segmentation-detection module. A ResNet-like feature extractor first extracts features, which are then upscaled and
average-pooled to capture neighboring context. Features are further refined for anomaly detection task in the adaptor module.
During training, synthetic anomalies are generated at the feature level by adding Gaussian noise to regions defined by the
binary Perlin noise mask. The perturbed features are then fed into the segmentation-detection
module, which produces the anomaly map and the anomaly score. During inference, anomaly generation is skipped, and the model
directly predicts the anomaly map and score. The predicted anomaly map is upscaled to match the input image size
and refined with a Gaussian filter.

This model can be trained in both unsupervised and supervised setting, but Anomalib currently supports only unsupervised training.

## Architecture

![SuperSimpleNet architecture](/docs/source/images/supersimplenet/architecture.png "SuperSimpleNet architecture")

## Usage

`anomalib train --model SuperSimpleNet --data MVTec --data.category <category>`

## Benchmark
