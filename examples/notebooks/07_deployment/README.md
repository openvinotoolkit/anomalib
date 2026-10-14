# Deployment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-edge-platform/anomalib/blob/main/examples/notebooks/07_deployment/openvino_quantization.ipynb)

This section contains notebooks that demonstrate model deployment and optimization techniques.

## 📚 Notebooks

| Notebook                                                     | Description                                          |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [`openvino_quantization.ipynb`](openvino_quantization.ipynb) | Model compression using NNCF for OpenVINO deployment |

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](https://anomalib.readthedocs.io/en/latest/markdown/get_started/anomalib.html#installation).

## Notebook Contents

This notebook demonstrates how NNCF can be used to compress a model trained with Anomalib. The notebook is divided into the following sections:

- Train an anomalib model without compression
- Train a model with NNCF compression
- Compare the performance of the two models (FP32 vs INT8)
