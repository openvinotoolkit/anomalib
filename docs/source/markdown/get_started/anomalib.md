# Anomalib in 15 Minutes

This section will walk you through the steps to train a model and use it to detect anomalies in a dataset.

## {octicon}`package` Installation

Installation is simple and can be done in two ways. The first is through PyPI, and the second is through a local installation. PyPI installation is recommended if you want to use the library without making any changes to the source code. If you want to make changes to the library, then a local installation is recommended.

:::::{dropdown} Installing the Installer
:open:

Anomalib comes with a CLI installer that can be used to install the full package.
The installer can be installed using the following commands:

::::{tab-set}

:::{tab-item} API
:sync: label-1

```{literalinclude} ../../snippets/install/pypi.txt
:language: bash
```

:::

:::{tab-item} Source
:sync: label-2

```{literalinclude} ../../snippets/install/source.txt
:language: bash
```

:::
::::
The main reason why PyPI and source installer does not install the full package
is to keep the installation wheel small. The CLI installer also automates the
installation such as finding the torch version with the right CUDA/CUDNN version.

The next section demonstrates how to install the full package using the CLI installer.
:::::

:::::{dropdown} Installing the Full Package
After installing anomalib, you can install the full package using the following commands:

```{literalinclude} ../../snippets/install/anomalib_help.txt
:language: bash
```

As can be seen above, the only available sub-command is `install` at the moment.
The `install` sub-command has options to install either the full package or the
specific components of the package.

```{literalinclude} ../../snippets/install/anomalib_install_help.txt
:language: bash
```

By default the `install` sub-command installs the full package. If you want to
install only the specific components of the package, you can use the `--option` flag.

```{literalinclude} ../../snippets/install/anomalib_install.txt
:language: bash
```

After following these steps, your environment will be ready to use anomalib!
:::::

## {octicon}`mortar-board` Training

Anomalib supports both API and CLI-based training. The API is more flexible and allows for more customization, while the CLI training utilizes command line interfaces, and might be easier for those who would like to use anomalib off-the-shelf.

::::{tab-set}

:::{tab-item} API

```{literalinclude} ../../snippets/train/api/default.txt
:language: python
```

:::

:::{tab-item} CLI

```{literalinclude} ../../snippets/train/cli/default.txt
:language: bash
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

```{literalinclude} ../../snippets/inference/api/lightning.txt
:language: python
```

:::

:::{tab-item} CLI
:sync: label-2

```{literalinclude} ../../snippets/inference/cli/lightning.txt
:language: bash
```

:::

::::
:::::

:::::{dropdown} Torch Inference

::::{tab-set}

:::{tab-item} API
:sync: label-1

```{code-block} python
Python code here.
```

:::

:::{tab-item} CLI
:sync: label-2

```{code-block} bash
CLI command here.
```

:::

::::
:::::

:::::{dropdown} OpenVINO Inference

::::{tab-set}

:::{tab-item} API
:sync: label-1

```{code-block} python
Python code here.
```

:::

:::{tab-item} CLI
:sync: label-2

```{code-block} bash
CLI command here.
```

:::

::::
:::::

:::::{dropdown} Gradio Inference

::::{tab-set}

:::{tab-item} API
:sync: label-1

```{code-block} python
Python code here.
```

:::

:::{tab-item} CLI
:sync: label-2

```{code-block} bash
CLI command here.
```

:::

::::
:::::

## {octicon}`graph` Hyper-Parameter Optimization

Anomalib supports hyper-parameter optimization using [wandb](https://wandb.ai/) and [comet.ml](https://www.comet.com/). Here we show an example of hyper-parameter optimization using both comet and wandb.

::::{tab-set}

:::{tab-item} CLI

```{literalinclude} ../../snippets/pipelines/hpo/cli.txt
:language: bash
```

:::

:::{tab-item} API

```{literalinclude} ../../snippets/pipelines/hpo/api.txt
:language: bash
```

:::

::::

## {octicon}`beaker` Experiment Management

Anomalib is integrated with various libraries for experiment tracking such as comet, tensorboard, and wandb through [lighting loggers](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html).

::::{tab-set}

:::{tab-item} CLI

To run a training experiment with experiment tracking, you will need the following configuration file:

```{code-block} yaml
# Place the experiment management config here.
```

By using the configuration file above, you can run the experiment with the following command:

```{literalinclude} ../../snippets/logging/cli.txt
:language: bash
```

:::

:::{tab-item} API

```{literalinclude} ../../snippets/logging/api.txt
:language: bash
```

:::

::::

## {octicon}`meter` Benchmarking

Anomalib provides a benchmarking tool to evaluate the performance of the anomaly detection models on a given dataset. The benchmarking tool can be used to evaluate the performance of the models on a given dataset, or to compare the performance of multiple models on a given dataset.

Each model in anomalib is benchmarked on a set of datasets, and the results are available in `src/anomalib/models/<model_name>README.md`. For example, the MVTec AD results for the Patchcore model are available in the corresponding [README.md](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/image/patchcore#mvtec-ad-dataset) file.

::::{tab-set}

:::{tab-item} CLI

To run the benchmarking tool, run the following command:

```{code-block} bash
anomalib benchmark --config tools/benchmarking/benchmark_params.yaml
```

:::

:::{tab-item} API

```{code-block} python
# To be enabled in v1.1
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
