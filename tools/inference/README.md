# Inference

These files provide an example on how to use the trained (and optionally optimized) models.

Currently, you can use the use the torch, OpenVINO, and lightning models.

## Lightning Inference

The `lightning_inference.py` provides an example on how to use the trained `ckpt` file to run inference on a dataset.

Example:

```bash
python tools/inference/lightning_inference.py \
  --model anomalib.models.Padim \
  --ckpt_path results/padim/mvtec/bottle/weights/lightning/model.ckpt \
  --data.path datasets/MVTec/bottle/test/broken_large \
  --output ./outputs
```

You can also use a config file with the entrypoint

Here is a simple YAML file for Padim Model.

```yaml
ckpt_path: results/padim/mvtec/bottle/weights/lightning/model.ckpt
data:
  path: datasets/MVTec/bottle/test/broken_large
  transform: null
  image_size:
    - 256
    - 256
output: ./outputs
visualization_mode: simple
show: false
model:
  class_path: anomalib.models.Padim
  init_args:
    layers:
      - layer1
      - layer2
      - layer3
    input_size:
      - 256
      - 256
    backbone: resnet18
    pre_trained: true
    n_features: null
```

You can then use

```bash
python tools/inference/lightning_inference.py -c config.yaml
```
