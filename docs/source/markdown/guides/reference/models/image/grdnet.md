# GRD-Net

GRD-Net is a one-class reconstruction and segmentation model. It denoises Perlin-corrupted tiles with a residual encoder-decoder-encoder generator and learns anomaly localization with the existing DRÆM segmentator. Optional ROI masks supervise training targets only; prediction and export accept images without ROI masks.

## Architecture

```{eval-rst}
.. image:: ../../../../../images/grdnet/architecture.svg
   :alt: GRD-Net training and inference
```

The default preprocessor resizes RGB input to 256 × 256 without normalization. Internal tiles are 128 × 128 with stride 64. Predictions average overlaps, smooth the full map with a 21 × 21 mean filter, and use its maximum as the image score.

## Usage and measured results

The [model README](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/grdnet/README.md) provides Python, CLI, and YAML examples, checkpoint/export usage, and selected-category MVTec AD measurements. Historical paper values, through-200 selections, the 300-epoch extension, and Hazelnut geometry sensitivity are reported separately. Image and raw-pixel checkpoints were independently selected using test labels; these are not unbiased generalization estimates. The public output map is smoothed, unlike the raw map used for pixel checkpoint selection.

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
