.. _benchmarking:

Benchmarking
=============

To add to the suit of experiment tracking and optimization, anomalib also includes a benchmarking script for gathering results across different combinations of models, their parameters, and dataset categories. The model performance and throughputs are logged into a csv file that can also serve as a means to track model drift. Optionally, these same results can be logged to Comet, Weights and Biases and TensorBoard. A sample configuration file is shown below.

.. code-block:: yaml

  seed: 42
  compute_openvino: false
  hardware:
    - cpu
    - gpu
  writer:
    - comet
    - wandb
    - tensorboard
  grid_search:
    dataset:
      category:
        - colour
        - crack
      image_size: [128, 256]
    model_name:
      - padim
      - stfpm

This configuration computes the throughput and performance metrics on CPU and GPU for two categories of a custom folder dataset for Padim and STFPM models. To configure a custom dataset, use the respective model configuration file. An example for dataset configuration used in this guide is shown below. Refer `README <https://github.com/openvinotoolkit/anomalib#readme>`_ for more details.

.. code-block:: yaml

  dataset:
    name: hazelnut
    format: folder
    path: path/hazelnut_toy
    normal_dir: good # name of the folder containing normal images.
    abnormal_dir: colour # name of the folder containing abnormal images.
    normal_test_dir: null
    task: segmentation # classification or segmentation
    mask: path/hazelnut_toy/mask/colour
    extensions: .jpg
    split_ratio: 0.2
    seed: 0
    image_size: 256

Additionally, it is possible to pass a single value instead of an array for any specific parameter. This will overwrite the parameter in each of the model configs and thereby ensures that the parameter is kept constant between all runs in the sweep. For example, to ensure that the same dataset is used between runs the configuration file can be modified as shown below.

.. code-block:: yaml

  seed: 42
  compute_openvino: false
  hardware:
    - cpu
    - gpu
  writer:
    - comet
    - wandb
    - tensorboard
  grid_search:
    dataset:
      name: hazelnut
      format: folder
      path: path/hazelnut_toy
      normal_dir: good # name of the folder containing normal images.
      abnormal_dir: colour # name of the folder containing abnormal images.
      normal_test_dir: null
      task: segmentation # classification or segmentation
      mask: path/hazelnut_toy/mask/colour
      extensions: .jpg
      split_ratio: 0.2
      category:
        - colour
        - crack
      image_size: [128, 256]
    model_name:
      - padim
      - stfpm

By default, ``compute_openvino`` is set to ``False`` to support instances where OpenVINO requirements are not installed in the environment. Once installed, this flag can be set to ``True`` to get the throughput on OpenVINO optimized models. The ``writer`` parameter is optional and can be set to ``writer: []`` in case the user only requires a csv file without logging to each respective logger. It is a good practice to set a value of seed to ensure reproducibility across runs and thus, is set to a non-zero value by default.

Once a configuration is decided, benchmarking can easily be performed by calling

.. code-block:: bash

  python tools/benchmarking/benchmark.py --config <relative/absolute path>/<paramfile>.yaml


A nice feature about the provided benchmarking script is that if the host system has multiple GPUs, the runs are parallelized over all the available GPUs for faster collection of result.
