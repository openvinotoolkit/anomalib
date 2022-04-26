# Benchmarking

To add to the suit of experiment tracking and optimization, anomalib also includes a benchmarking script for gathering results across different combinations of models, their parameters, and dataset categories. The model performance and throughputs are logged into a csv file that can also serve as a means to track model drift. Optionally, these same results can be logged to Weights and Biases and TensorBoard. A sample configuration file is shown below.

```yaml
seed: 42
compute_openvino: false
hardware:
  - cpu
  - gpu
writer:
  - wandb
  - tensorboard
grid_search:
  dataset:
    category:
      - bottle
      - cable
      - capsule
      - carpet
    image_size: [224]
  model_name:
    - padim
    - patchcore
```

This configuration computes the throughput and performance metrics on CPU and GPU for four categories of the MVTec dataset for Padim and PatchCore models. The dataset can be configured in the respective model configuration files. By default, `compute_openvino` is set to `False` to support instances where OpenVINO requirements are not installed in the environment. Once installed, this flag can be set to True to get throughput on OpenVINO optimized models. The `writer` parameter is optional and can be set to `writer: []` in case the user only requires a csv file without logging to each respective logger. It is a good practice to set a value of seed to ensure reproducibility across runs and thus, is set to a non-zero value by default.

Once a configuration is decided, benchmarking can easily be performed by calling
```bash
python tools/benchmarking/benchmark.py --config <relative/absolute path>/<paramfile>.yaml
```
A nice feature about the provided benchmarking script is that if the host system has multiple GPUs, the runs are parallelized over all the available GPUs for faster collection of result.
