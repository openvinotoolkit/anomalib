# Anomalib in 15 Minutes

This section will walk you through the steps to train a model and use it to detect anomalies in a dataset.

## {octicon}`package` Installation

Anomalib can be installed from PyPI. We recommend using a virtual environment and a modern package installer like `uv` or `pip`.

### Quick Install

For a standard installation, you can use `uv` or `pip`. This will install the latest version of Anomalib with its core dependencies. PyTorch will be installed based on its default behavior, which usually works for CPU and standard CUDA setups.

```bash
# With uv
uv pip install anomalib

# Or with pip
pip install anomalib
```

For more control over the installation, such as specifying the PyTorch backend or installing extra dependencies, see the advanced options below.

:::::{dropdown} Advanced: Install with Hardware-Specific Extras

To ensure compatibility with your hardware, you can specify a backend during installation. This is the recommended approach for production environments and for hardware other than CPU or standard CUDA.

**Using `uv`:**

```{literalinclude} ../../../../examples/cli/00_installation/uv_install.sh
:language: bash
:lines: 10-30
```

**Using `pip`:**

```{literalinclude} ../../../../examples/cli/00_installation/pip_install.sh
:language: bash
:lines: 9-17
```

:::::

## {octicon}`mortar-board` Training

Anomalib supports both API and CLI-based training. The API is more flexible
and allows for more customization, while the CLI training utilizes command line
interfaces, and might be easier for those who would like to use anomalib off-the-shelf.

::::{tab-set}

:::{tab-item} API

```{literalinclude} ../../../../examples/api/01_getting_started/basic_training.py
:language: python
:lines: 10-48
```

:::

:::{tab-item} CLI

```{literalinclude} ../../../../examples/cli/01_getting_started/basic_training.sh
:language: bash
:lines: 10-33
```

:::

::::

## {octicon}`cpu` Inference

Anomalib includes multiple inferencing scripts, including Torch, Lightning, Gradio, and OpenVINO inferencers to perform inference using the trained/exported model. Here we show an inference example using the Lightning inferencer.

:::::{dropdown} Lightning Inference
:open:

::::{tab-set}

:::{tab-item} API
:sync: label-1

```{literalinclude} ../../../../examples/api/01_getting_started/basic_inference.py
:language: python
:lines: 10-39
```

:::

:::{tab-item} CLI
:sync: label-2

```{literalinclude} ../../../../examples/cli/01_getting_started/basic_inference.sh
:language: bash
```

:::

::::
:::::

:::::{dropdown} Torch Inference

::::{tab-set}

:::{tab-item} API
:sync: label-1

```{literalinclude} ../../../../examples/api/01_getting_started/basic_torch_inference.py
:language: python
:lines: 10-11
```

:::

:::{tab-item} CLI
:sync: label-2

```{code-block} bash

```

:::

::::
:::::

:::::{dropdown} OpenVINO Inference

Exported OpenVINO models can be run using the optimized `OpenVINOInferencer`:

::::{tab-set}

:::{tab-item} API
:sync: label-1

```{literalinclude} ../../../../examples/api/01_getting_started/basic_openvino_inference.py
:language: python
:lines: 10-27
```

:::

:::{tab-item} CLI
:sync: label-2

```bash
anomalib predict \
    --model Patchcore \
    --data anomalib.data.MVTecAD \
    --ckpt_path results/Patchcore/MVTecAD/weights/openvino/model.bin
```

:::

::::
:::::

## {octicon}`beaker` Experiment Management

Anomalib integrates with popular experiment tracking platforms such as TensorBoard, Weights & Biases, and MLflow through Lightning loggers.

::::{tab-set}

:::{tab-item} CLI

To run training with experiment tracking enabled:

```{literalinclude} ../../snippets/logging/cli.txt
:language: bash
```

Or configure via YAML:

```{literalinclude} ../../snippets/logging/config.txt
:language: yaml
```

:::

:::{tab-item} API

```{literalinclude} ../../snippets/logging/api.txt
:language: python
```

:::

::::

## {octicon}`meter` Benchmarking

Anomalib provides a benchmarking tool to evaluate the performance of the anomaly detection models on a given dataset. The benchmarking tool can be used to evaluate the performance of the models on a given dataset, or to compare the performance of multiple models on a given dataset.

Each model in anomalib is benchmarked on a set of datasets, and the results are available in `src/anomalib/models/<model_name>README.md`. For example, the MVTecAD AD results for the Patchcore model are available in the corresponding [README.md](https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/patchcore#mvtec-ad-dataset) file.

::::{tab-set}

:::{tab-item} CLI

To run the benchmarking tool, run the following command:

```{literalinclude} ../../snippets/pipelines/benchmark/cli_anomalib.txt
:language: bash
```

:::

:::{tab-item} API

```python
from anomalib.pipelines.benchmark import Benchmark

benchmark = Benchmark(config="tools/experimental/benchmarking/sample.yaml")
benchmark.run()
```

:::

::::

## {octicon}`bookmark` Reference

If you use this library and love it, use this to cite it:

```{code-block} bibtex
@inproceedings{akcay2022anomalib,
  title={Anomalib: A deep learning library for anomaly detection},
  author={
    Akcay, Samet and
    Ameln, Dick and
    Vaidya, Ashwin and
    Lakshmanan, Barath
    and Ahuja, Nilesh
    and Genc, Utku
  },
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={1706--1710},
  year={2022},
  organization={IEEE}
}
```
