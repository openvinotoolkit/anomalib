# Video Datamodules

Video datamodules in Anomalib are designed to handle video-based anomaly detection datasets. They provide a standardized interface for loading and processing video data for both training and inference.

## Available Datamodules

```{grid} 3
:gutter: 2

:::{grid-item-card} Avenue
:link: anomalib.data.Avenue
:link-type: doc

CUHK Avenue dataset for video anomaly detection.
:::

:::{grid-item-card} ShanghaiTech
:link: anomalib.data.ShanghaiTech
:link-type: doc

ShanghaiTech dataset for video anomaly detection.
:::

:::{grid-item-card} UCSDped
:link: anomalib.data.UCSDped
:link-type: doc

UCSD Pedestrian dataset for video anomaly detection.
:::
```

## API Reference

```{eval-rst}
.. automodule:: anomalib.data
   :members: Avenue, ShanghaiTech, UCSDped
   :undoc-members:
   :show-inheritance:
```
