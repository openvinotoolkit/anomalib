# Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection

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

![CS-Flow Architecture](/docs/source/images/cs_flow/architecture1.jpg "CS-Flow Architecture")

![Architecture of a Coupling Block](/docs/source/images/cs_flow/architecture2.jpg "Architecture of a Coupling Block")

![Architecture of network predicting scale and shift parameters.](/docs/source/images/cs_flow/architecture3.jpg "Architecture of network predicting scale and shift parameters.")

## Usage

`anomalib train --model Csflow --data MVTec --data.category <category>`

## Benchmark

All results gathered with seed `42`.

## [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

> The following table is generated with image size of 768 and generating the anomaly map from all the three scales unlike the paper. Initial experiments showed that the anomaly map from all the three scales gives better results than the one from the largest scale.

### Image AUROC - 768 Image Size

|                 | Average | Carpet |  Grid | Leather |  Tile |  Wood | Bottle | Cable | Capsule | Hazelnut | Metal_nut | Pill | Screw | Toothbrush | Transistor | Zipper |
| :-------------- | ------: | -----: | ----: | ------: | ----: | ----: | -----: | ----: | ------: | -------: | --------: | ---: | ----: | ---------: | ---------: | -----: |
| EfficientNet-B5 |   0.987 |      1 | 0.989 |       1 | 0.998 | 0.998 |      1 | 0.996 |   0.981 |    0.994 |         1 | 0.98 |  0.95 |      0.919 |          1 |  0.999 |

### Pixel AUROC - 768 Image Size

|                 | Average | Carpet |  Grid | Leather |  Tile |  Wood | Bottle | Cable | Capsule | Hazelnut | Metal_nut | Pill | Screw | Toothbrush | Transistor | Zipper |
| :-------------- | ------: | -----: | ----: | ------: | ----: | ----: | -----: | ----: | ------: | -------: | --------: | ---: | ----: | ---------: | ---------: | -----: |
| EfficientNet-B5 |   0.921 |  0.936 | 0.878 |   0.917 | 0.872 | 0.782 |  0.889 | 0.935 |   0.961 |    0.957 |     0.953 | 0.95 | 0.947 |      0.951 |      0.974 |  0.919 |

### Pixel F1Score - 768 Image Size

|                 | Average | Carpet |  Grid | Leather | Tile |  Wood | Bottle | Cable | Capsule | Hazelnut | Metal_nut |  Pill | Screw | Toothbrush | Transistor | Zipper |
| :-------------- | ------: | -----: | ----: | ------: | ---: | ----: | -----: | ----: | ------: | -------: | --------: | ----: | ----: | ---------: | ---------: | -----: |
| EfficientNet-B5 |    0.33 |  0.219 | 0.104 |   0.144 | 0.41 | 0.211 |  0.357 | 0.375 |   0.333 |    0.375 |     0.689 | 0.458 | 0.094 |      0.342 |      0.597 |  0.238 |

### Image F1 Score - 768 Image Size

|                 | Average | Carpet |  Grid | Leather |  Tile |  Wood | Bottle | Cable | Capsule | Hazelnut | Metal_nut |  Pill | Screw | Toothbrush | Transistor | Zipper |
| :-------------- | ------: | -----: | ----: | ------: | ----: | ----: | -----: | ----: | ------: | -------: | --------: | ----: | ----: | ---------: | ---------: | -----: |
| EfficientNet-B5 |   0.985 |      1 | 0.991 |       1 | 0.988 | 0.992 |      1 | 0.973 |   0.977 |    0.979 |     0.995 | 0.975 | 0.975 |      0.952 |      0.988 |  0.996 |

> For fair comparison with other algorithms, the following results are computed with image size of 256.

### Image AUROC - 256 Image Size

|                 | Average | Carpet |  Grid | Leather |  Tile |  Wood | Bottle | Cable | Capsule | Hazelnut | Metal_nut |  Pill | Screw | Toothbrush | Transistor | Zipper |
| :-------------- | ------: | -----: | ----: | ------: | ----: | ----: | -----: | ----: | ------: | -------: | --------: | ----: | ----: | ---------: | ---------: | -----: |
| EfficientNet-B5 |   0.972 |  0.995 | 0.982 |       1 | 0.972 | 0.988 |      1 |  0.97 |   0.907 |    0.995 |     0.972 | 0.953 | 0.896 |      0.969 |      0.987 |  0.987 |

### Pixel AUROC - 256 Image Size

|                 | Average | Carpet |  Grid | Leather |  Tile |  Wood | Bottle | Cable | Capsule | Hazelnut | Metal_nut | Pill | Screw | Toothbrush | Transistor | Zipper |
| :-------------- | ------: | -----: | ----: | ------: | ----: | ----: | -----: | ----: | ------: | -------: | --------: | ---: | ----: | ---------: | ---------: | -----: |
| EfficientNet B5 |   0.845 |  0.847 | 0.746 |   0.851 | 0.775 | 0.677 |  0.853 | 0.863 |   0.882 |    0.895 |     0.932 | 0.92 | 0.779 |      0.892 |       0.96 |  0.803 |

### Pixel F1Score - 256 Image Size

|                 | Average | Carpet |  Grid | Leather |  Tile |  Wood | Bottle | Cable | Capsule | Hazelnut | Metal_nut |  Pill | Screw | Toothbrush | Transistor | Zipper |
| :-------------- | ------: | -----: | ----: | ------: | ----: | ----: | -----: | ----: | ------: | -------: | --------: | ----: | ----: | ---------: | ---------: | -----: |
| EfficientNet B5 |   0.231 |  0.108 | 0.069 |   0.048 | 0.306 | 0.127 |  0.303 |  0.21 |   0.165 |    0.215 |     0.659 | 0.412 | 0.017 |      0.214 |      0.513 |  0.106 |

### Image F1 Score - 256 Image Size

|                 | Average | Carpet |  Grid | Leather |  Tile |  Wood | Bottle | Cable | Capsule | Hazelnut | Metal_nut |  Pill | Screw | Toothbrush | Transistor | Zipper |
| :-------------- | ------: | -----: | ----: | ------: | ----: | ----: | -----: | ----: | ------: | -------: | --------: | ----: | ----: | ---------: | ---------: | -----: |
| EfficientNet B5 |   0.965 |  0.983 | 0.982 |       1 | 0.957 | 0.966 |      1 | 0.945 |   0.944 |    0.986 |     0.963 | 0.965 | 0.906 |      0.949 |      0.938 |  0.987 |

### Sample Results

### TODO: Add results
