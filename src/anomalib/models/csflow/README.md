# Real-Time Unsupervised Anomaly Detection via Conditional Normalizing Flows

This is the implementation of the [CS-Flow](https://arxiv.org/pdf/2110.02855.pdf) paper. This code is modified form of the [official repository](https://github.com/marco-rudolph/cs-flow).

Model Type: Segmentation

## Description

The central idea of the paper is to handle fine-grained representations by incorporating global and local image context. This is done by taking multiple scales when extracting features and using a fully-convolutional normalizing flow to process the scales jointly. This can be seen in Figure 1.

In each cross-scale coupling block, the input tensor is split into two parts across the channel dimension. Similar to RealNVP, each part is used to compute the scale and translate parameters for the affine transform. This is done with the help of cross-scale convolution layers as shown in Figure 2. These are point wise operations. As shown in the figure, the subnetworks are $r_1$ and $r_2$ and their outputs are $[s_1, t_1]$ and $[s_2, t_2]$. Then, the output of the coupling blocks are defined as.

$$
y_{out,2} = y_{in,2} \odot e^{\gamma_1s_1(y_{in,1}) + \gamma_1t_1(y_{in,1})}\\
y_{out,1} = y_{in,1} \odot e^{\gamma_2s_2(y_{out,2}) + \gamma_2t_2(y_{out,2})}
$$

Here, $\gamma_1$ and $\gamma_2$ are learnable parameters for each block.

Figure 3 shows the architecture of the subnetworks in detail.

The anomaly score for each local position $(i,j)$ of the feature map $y^s$ at scale $s$ is computed by aggregating values along the channel dimension with $||z^s_{i,j}||^2_2$. Here $z$ is the latent variable and $z^s_{i,j}$ is the output of the final coupling block at scale $s$ for the local position $(i,j)$. Thus anomalies can be localized by marking image regions with high norm in output feature tensors $z^s$.

## Architecture

![CS-Flow Architecture](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/cs_flow/architecture1.jpg "CS-Flow Architecture")

![Architecture of a Coupling Block](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/cs_flow/architecture2.jpg "Architecture of a Coupling Block")

![Architecture of network predicting scale and shift parameters.](https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/cs_flow/architecture3.jpg "Architecture of network predicting scale and shift parameters.")

## Usage

`python tools/train.py --model cs_flow`

## Benchmark

All results gathered with seed `42`.

## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

| Model      | Image AUROC | Image F1 | Pixel AUROC | Pixel AUPRO |
| ---------- | ----------- | -------- | ----------- | ----------- |
| Bottle     |             |          |             |             |
| Cable      |             |          |             |             |
| Capsule    |             |          |             |             |
| Carpet     |             |          |             |             |
| Grid       |             |          |             |             |
| Hazelnut   |             |          |             |             |
| Leather    |             |          |             |             |
| Metal_nut  |             |          |             |             |
| Pill       |             |          |             |             |
| Screw      |             |          |             |             |
| Tile       |             |          |             |             |
| Toothbrush |             |          |             |             |
| Transistor |             |          |             |             |
| Wood       |             |          |             |             |
| Zipper     |             |          |             |             |
| Average    |             |          |             |             |

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
| candle     |             |          |             |             |
| capsules   |             |          |             |             |
| cashew     |             |          |             |             |
| chewinggum |             |          |             |             |
| fryum      |             |          |             |             |
| macaroni1  |             |          |             |             |
| macaroni2  |             |          |             |             |
| pcb1       |             |          |             |             |
| pcb2       |             |          |             |             |
| pcb3       |             |          |             |             |
| pcb4       |             |          |             |             |
| pipe_fryum |             |          |             |             |
| Average    |             |          |             |             |
