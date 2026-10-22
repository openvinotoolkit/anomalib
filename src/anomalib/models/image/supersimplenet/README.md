# SuperSimpleNet

This is an implementation of the SuperSimpleNet, based on the [official code](https://github.com/blaz-r/SuperSimpleNet).

The model was first presented at ICPR 2024: [SuperSimpleNet : Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection](https://arxiv.org/abs/2408.03143)

An extension was later published in JIMS 2025: [No Label Left Behind: A Unified Surface Defect Detection Model for all Supervision Regimes](https://link.springer.com/article/10.1007/s10845-025-02680-8)

Model Type: Segmentation

## Description

**SuperSimpleNet** is a simple yet strong discriminative defect / anomaly detection model evolved from the SimpleNet architecture. It consists of four components:
feature extractor with upscaling, feature adaptor, feature-level synthetic anomaly generation module, and
segmentation-detection module.

A ResNet-like feature extractor first extracts features, which are then upscaled and
average-pooled to capture neighboring context. Features are (optionally) further refined for anomaly detection task in the adaptor module.
During training, synthetic anomalies are generated at the feature level by adding Gaussian noise to regions defined by the
binary Perlin noise mask. The perturbed features are then fed into the segmentation-detection
module, which produces the anomaly map and the anomaly score. During inference, anomaly generation is skipped, and the model
directly predicts the anomaly map and score. The predicted anomaly map is upscaled to match the input image size
and refined with a Gaussian filter.

This implementation supports both unsupervised and supervised setting, but Anomalib currently supports only unsupervised learning.

## Architecture

![SuperSimpleNet architecture](/docs/source/images/supersimplenet/architecture.png "SuperSimpleNet architecture")

Currently, the difference between ICPR and JIMS code is only the `adapt_cls_features` which controls whether the features used for classification head are adapted or not.
For ICPR this is set to True (i.e. the features for classification head are adapted), and for JIMS version this is False (which is also the default).

## Usage

`anomalib train --model Supersimplenet --data MVTecAD --data.category <category>`

> IMPORTANT!
>
> The model is verified to work with WideResNet50 using torchvision V1 weights.
> It should work with most ResNets and WideResNets, but make sure you use V1 weights if you use default noise std value.
> Correct weight name ends with ".tv\_[...]", not "tv2" (e.g. "wide_resnet50_2.tv_in1k").
>
> It is recommended to train the model for 300 epochs with batch size of 32 to achieve stable training with random anomaly generation. Training with lower parameter values will still work, but might not yield the optimal results.
>
> For weakly, mixed and fully supervised training, refer to the [official code](https://github.com/blaz-r/SuperSimpleNet).

## MVTecAD AD results

The following results were obtained using this Anomalib implementation trained for 300 epochs with seed 0, default params, and batch size 32.

| Category    | AUROC (ICPR) | AUROC (JIMS) | AUPRO (ICPR) | AUPRO (JIMS) |
| ----------- | :----------: | :----------: | :----------: | :----------: |
| Bottle      |    1.000     |    1.000     |    0.903     |    0.911     |
| Cable       |    0.981     |    0.951     |    0.901     |    0.893     |
| Capsule     |    0.989     |    0.992     |    0.931     |    0.919     |
| Carpet      |    0.985     |    0.974     |    0.929     |    0.935     |
| Grid        |    0.994     |    0.998     |    0.930     |    0.938     |
| Hazelnut    |    0.994     |    0.999     |    0.943     |    0.939     |
| Leather     |    1.000     |    1.000     |    0.970     |    0.974     |
| Metal_nut   |    0.995     |    0.993     |    0.920     |    0.925     |
| Pill        |    0.962     |    0.980     |    0.936     |    0.943     |
| Screw       |    0.912     |    0.854     |    0.947     |    0.946     |
| Tile        |    0.994     |    0.992     |    0.854     |    0.825     |
| Toothbrush  |    0.908     |    0.908     |    0.860     |    0.854     |
| Transistor  |    1.000     |    1.000     |    0.907     |    0.916     |
| Wood        |    0.987     |    0.991     |    0.858     |    0.872     |
| Zipper      |    0.995     |    0.999     |    0.928     |    0.944     |
| **Average** |  **0.980**   |  **0.975**   |  **0.914**   |  **0.916**   |

For other results on VisA, SensumSODF, and KSDD2, refer to the [paper](https://link.springer.com/article/10.1007/s10845-025-02680-8).
