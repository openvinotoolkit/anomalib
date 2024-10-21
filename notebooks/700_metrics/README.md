# AUPIMO Tutorials

| Notebook                                        | GitHub                                                                            | Colab                                                                                                                                                                                                          |
| ----------------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AUPIMO basics                                   | [701a_aupimo](/notebooks/700_metrics/701a_aupimo.ipynb)                           | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/anomalib/blob/main/notebooks/700_metrics/701a_aupimo.ipynb)              |
| AUPIMO representative samples and visualization | [701b_aupimo_advanced_i](/notebooks/700_metrics/701b_aupimo_advanced_i.ipynb)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/anomalib/blob/main/notebooks/700_metrics/701b_aupimo_advanced_i.ipynb)   |
| PIMO curve and integration bounds               | [701c_aupimo_advanced_ii](/notebooks/700_metrics/701c_aupimo_advanced_ii.ipynb)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/anomalib/blob/main/notebooks/700_metrics/701c_aupimo_advanced_ii.ipynb)  |
| (AU)PIMO of a random model                      | [701d_aupimo_advanced_iii](/notebooks/700_metrics/701d_aupimo_advanced_iii.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/anomalib/blob/main/notebooks/700_metrics/701d_aupimo_advanced_iii.ipynb) |
| AUPIMO load/save, statistical comparison        | [701e_aupimo_advanced_iv](/notebooks/700_metrics/701e_aupimo_advanced_iv.ipynb)   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/anomalib/blob/main/notebooks/700_metrics/701e_aupimo_advanced_iv.ipynb)  |

## Installation Instructions

If you have not installed all required dependencies, follow the [Installation Guide](https://openvinotoolkit.github.io/anomalib/getting_started/installation/index.html).

## Notebook Contents

### AUPIMO (701 series)

The first notebook (`701a_aupimo`) introduces the basic usages of AUPIMO. It covers how to:

- get the average AUPIMO in the `Engine`
- access all individual AUPIMO values using the torchmetrics API.

The following notebooks show other advanced usages of AUPIMO, including how to:

- (in `701b_aupimo_advanced_i`)
  - select representative anomalous samples from the distribution of AUPIMO scores
  - visualize AUPIMO with heatmaps
- (in `701c_aupimo_advanced_ii`)
  - plot the PIMO curve and the integration bounds
  - visualize the validation condition imposed by the integration bounds (i.e. what the FPRn bounds mean in practice)
- (in `701d_aupimo_advanced_iii`)
  - a reference value for AUPIMO, calculated from a model that predicts random anomaly scores
- (in `701e_aupimo_advanced_iv`)
  - TODO(jpcbertoldo): add description
