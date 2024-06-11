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

`anomalib train --model EfficientAd --data anomalib.data.MVTec --data.category <category> --data.train_batch_size 1`

## Benchmark

All results gathered with seed `42`.

## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Image-Level AUC

|               |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| ------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| EfficientAd-S | 0.982 | 0.982  | 1.000 |  0.997  | 1.000 | 0.986 | 1.000  | 0.952 |  0.950  |  0.952   |   0.979   | 0.987 | 0.960 |   0.997    |   0.999    | 0.994  |
| EfficientAd-M | 0.975 | 0.972  | 0.998 |  1.000  | 0.999 | 0.984 | 0.991  | 0.945 |  0.957  |  0.948   |   0.989   | 0.926 | 0.975 |   1.000    |   0.965    | 0.971  |

### Image F1 Score

|               |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| ------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| EfficientAd-S | 0.970 | 0.966  | 1.000 |  0.995  | 1.000 | 0.975 | 1.000  | 0.907 |  0.956  |  0.897   |   0.978   | 0.982 | 0.944 |   0.984    |   0.988    | 0.983  |
| EfficientAd-M | 0.966 | 0.977  | 0.991 |  1.000  | 0.994 | 0.967 | 0.984  | 0.922 |  0.969  |  0.884   |   0.984   | 0.952 | 0.955 |   1.000    |   0.929    | 0.979  |
