# WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation

This is the implementation of the [WinCLIP](https://arxiv.org/pdf/2303.14814.pdf) paper.

Model Type: Segmentation

## Description

WinCLIP is a zero-shot/few-shot model for anomaly classification and segmentation. WinCLIP uses a pre-trained [CLIP](https://arxiv.org/pdf/2210.08901.pdf) model to extract image embeddings from the input images, and text embeddings from a set of pre-defined prompts describing the normal and anomalous states of the object class (e.g. "transistor without defect", "transistor with defect"). The image-level anomaly scores are obtained by computing the cosine similarity between the image embeddings and the normal and anomalous text embeddings.

In addition, WinCLIP performs pixel-level anomaly localization by repeating the anomaly score computation for different local regions of the image. This is achieved by moving a mask over the image in a sliding window fashion. The size of the mask can be varied to include different scales in the localization predictions. The similarity scores of the masked image is assigned to all the pixels in the masked region, after which the scores are aggregated across scales and window locations using harmonic averaging.

In few-shot mode, a reference association module is introduced, which collects and stores the (window-based) image embeddings of a selection of normal reference images. During inference, an additional association score is computed between as the cosine similarity between the embeddings of the input images and the normal reference images. The final anomaly score is the average of the zero-shot anomaly score and the few-shot association score.

## Architecture

![WinCLIP Architecture](/docs/source/images/winclip/architecture.png "WinCLIP Architecture")

## Usage

WinCLIP is a zero-shot model, which means that we can directly evaluate the model on a test set without training or fine-tuning on normal images.

### 0-Shot

`anomalib test --model WinClip --data MVTec`

### 1-Shot

`anomalib test --model WinClip --model.k_shot  1 --data MVTec`

## Parameters

| Parameter  | Type  | Description                                                                                                                                                   | Default  |
| :--------- | :---- | :------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------- |
| class_name | str   | Class name used in the prompt ensemble. When left empty, the category name from the dataset will be used if available, otherwise it will default to `object`. | `null`   |
| k_shot     | int   | Number of normal reference images used in few-shot mode.                                                                                                      | `0`      |
| scales     | tuple | Scales to be included in the multiscale window-embeddings. Each scale is an integer which indicates the window size in number of patches.                     | `[2, 3]` |

## Benchmark

Coming soon...

<!-- All results gathered with seed `42`. -->

<!-- ## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) -->

<!-- ### Image-Level AUC -->

<!-- |                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| 0-shot         |       |        |       |         |       |       |        |       |         |          |           |       |       |            |            |        |
| 1-shot         |       |        |       |         |       |       |        |       |         |          |           |       |       |            |            |        |
| 2-shot         |       |        |       |         |       |       |        |       |         |          |           |       |       |            |            |        |
| 4-shot         |       |        |       |         |       |       |        |       |         |          |           |       |       |            |            |        |

### Pixel-Level AUC

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| 0-shot         |       |        |       |         |       |       |        |       |         |          |           |       |       |            |            |        |
| 1-shot         |       |        |       |         |       |       |        |       |         |          |           |       |       |            |            |        |
| 2-shot         |       |        |       |         |       |       |        |       |         |          |           |       |       |            |            |        |
| 4-shot         |       |        |       |         |       |       |        |       |         |          |           |       |       |            |            |        |

### Image F1 Score

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| 0-shot         |       |        |       |         |       |       |        |       |         |          |           |       |       |            |            |        |
| 1-shot         |       |        |       |         |       |       |        |       |         |          |           |       |       |            |            |        |
| 2-shot         |       |        |       |         |       |       |        |       |         |          |           |       |       |            |            |        |
| 4-shot         |       |        |       |         |       |       |        |       |         |          |           |       |       |            |            |        | -->

<!-- ### Sample Results -->

## Attribution

The implementation of the torch model was inspired by https://github.com/zqhang/WinCLIP-pytorch and https://github.com/caoyunkang/WinClip.
