# DSR

This is the implementation of the [DSR](https://link.springer.com/chapter/10.1007/978-3-031-19821-2_31) paper.

Model Type: Segmentation

## Description

DSR is a quantized-feature based algorithm that consists of an autoencoder with one encoder and two decoders, coupled with an anomaly detection module. DSR learns a codebook of quantized representations on ImageNet, which are then used to encode input images. These quantized representations also serve to sample near-in-distribution anomalies, since they do not rely on external datasets. Training takes place in three phases. The encoder and "general object decoder", as well as the codebook, are pretrained on ImageNet. Defects are then generated at the feature level using the codebook on the quantized representations, and are used to train the object-specific decoder as well as the anomaly detection module. In the final phase of training, the upsampling module is trained on simulated image-level smudges in order to output more robust anomaly maps.

## Architecture

```{eval-rst}
.. image:: https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/dsr/architecture.png
    :alt: DSR Architecture
```

```{eval-rst}
.. automodule:: anomalib.models.image.dsr.torch_model
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: anomalib.models.image.dsr.lightning_model
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: anomalib.models.image.dsr.anomaly_generator
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: anomalib.models.image.dsr.loss
   :members:
   :undoc-members:
   :show-inheritance:
```
