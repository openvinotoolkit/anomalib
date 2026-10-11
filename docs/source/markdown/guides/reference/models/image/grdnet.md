# GRD-Net

GRD-Net is a one-class reconstruction and segmentation model. It denoises Perlin-corrupted tiles with a residual encoder-decoder-encoder generator and learns anomaly localization with the existing DRÆM segmentator. Optional ROI masks supervise training targets only; prediction and export accept images without ROI masks.

## Architecture

```{eval-rst}
.. image:: ../../../../../images/grdnet/architecture.svg
   :alt: GRD-Net training and inference
```

The default preprocessor resizes RGB input to 256 × 256 without normalization. Internal tiles are 128 × 128 with stride 64. Predictions average overlaps, smooth the full map with a 21 × 21 mean filter by default, and use the returned map's maximum as the image score.

## Configuration

Six keyword-only model options control contextual L1 and SSIM coefficients (both default `1.0`), generator scheduler decay (`0.1`) and patience (`3`), paired rotation (`90.0` degrees), and inference smoothing (`21`). Architecture and tile geometry remain fixed. Set these through the Python constructor, YAML `model.init_args`, or CLI `--model.<parameter>` arguments.

The scheduler reduction factor is `exp(-generator_scheduler_decay)`, not the decay value itself. It monitors contextual loss after the inner coefficients but before the outer `contextual_weight`; its absolute threshold stays `1e-4`. Changing inner coefficients therefore also changes the monitor's scale. Rotation zero preserves geometry and kernel one disables smoothing. A changed smoothing kernel requires renewed validation and calibration.

Checkpoint loading reconstructs these scalar options; older checkpoints use defaults. Resume with a checkpoint-reconstructed model or matching constructor settings, since `Engine.fit(..., ckpt_path=...)` alone does not reconstruct them. Restored optimizer/scheduler state takes precedence during resume, and a bare tensor `state_dict` does not store scalar options. See the model README for validation rules and examples.

## Usage and measured results

The [model README](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/grdnet/README.md) provides Python, CLI, and YAML examples, checkpoint/export usage, and selected-category MVTec AD measurements using the six option defaults. Nondefault configurations have not been benchmarked. Historical paper values, through-200 selections, the 300-epoch extension, and Hazelnut geometry sensitivity are reported separately. Image and raw-pixel checkpoints were independently selected using test labels; these are not unbiased generalization estimates. The public output map is smoothed by default, unlike the raw map used for pixel checkpoint selection.

For ROI-aware folders, see {doc}`../../data/datamodules/image/grdnet`. Standard anomalib datamodules need no ROI configuration.

## Sample results

These completed-run examples use the through-200 raw-pixel-selected checkpoints and display smoothed maps. Heatmaps are scaled for display only.

```{eval-rst}
.. image:: ../../../../../images/grdnet/results/0.png
   :alt: Hazelnut GRD-Net prediction

.. image:: ../../../../../images/grdnet/results/1.png
   :alt: Metal Nut GRD-Net prediction

.. image:: ../../../../../images/grdnet/results/2.png
   :alt: Pill GRD-Net prediction
```

The MVTec Software GmbH dataset-derived panels are [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), with a [separate asset notice](https://github.com/open-edge-platform/anomalib/blob/main/docs/source/images/grdnet/results/LICENSE).

## API

```{eval-rst}
.. automodule:: anomalib.models.image.grdnet.lightning_model
   :members: GRDNet
   :show-inheritance:
```

```{eval-rst}
.. automodule:: anomalib.models.image.grdnet.torch_model
   :members: GRDNetModel
   :show-inheritance:
```

```{eval-rst}
.. automodule:: anomalib.models.image.grdnet.loss
   :members: GRDNetGeneratorLoss, GRDNetDiscriminatorLoss, GRDNetSegmentatorLoss
   :show-inheritance:
```
