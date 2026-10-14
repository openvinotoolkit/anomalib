# Command Line Interface (CLI)

Anomalib provides a unified command line interface (`anomalib`) built on top of LightningCLI and `jsonargparse`.

## Commands Overview

The CLI provides the following subcommands:

- **`anomalib train`**: Fit and test a model on a dataset in a single run.
- **`anomalib fit`**: Train a model using the Lightning `Trainer.fit` lifecycle.
- **`anomalib validate`**: Validate a trained model checkpoint on the validation split.
- **`anomalib test`**: Evaluate a model checkpoint on the test split.
- **`anomalib predict`**: Run batch or single-image inference and generate anomaly maps.
- **`anomalib export`**: Export trained PyTorch checkpoints to OpenVINO IR or TorchScript.
- **`anomalib install`**: Manage extra dependencies and hardware accelerator packages.
- **`anomalib benchmark`**: Run benchmarking sweeps across models and datasets.

## Examples

### Train a model

```bash
anomalib train --model Patchcore --data MVTecAD --data.category bottle
```

### Run inference

```bash
anomalib predict \
    --ckpt_path results/Patchcore/MVTecAD/weights/lightning/model.ckpt \
    --data_path path/to/images
```

### Export a model

```bash
anomalib export \
    --model Patchcore \
    --export_type openvino \
    --ckpt_path results/Patchcore/MVTecAD/weights/lightning/model.ckpt
```

## API Reference

```{eval-rst}
.. automodule:: anomalib.cli.cli
   :members:
   :show-inheritance:
```
