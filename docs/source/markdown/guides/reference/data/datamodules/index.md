# Datamodules

Anomalib provides various datamodules for different types of data modalities. These datamodules are organized into three main categories:

## Image Datamodules

```{grid} 3
:gutter: 2

:::{grid-item-card} BTech
:link: image/btech
:link-type: doc

BTech dataset datamodule for surface defect detection.
:::

:::{grid-item-card} Datumaro
:link: image/datumaro
:link-type: doc

Datumaro format datamodule (compatible with Intel Geti™).
:::

:::{grid-item-card} Folder
:link: image/folder
:link-type: doc

Custom folder-based datamodule for organizing your own image dataset.
:::

:::{grid-item-card} Kolektor
:link: image/kolektor
:link-type: doc

Kolektor Surface-Defect dataset datamodule.
:::

:::{grid-item-card} MVTecAD
:link: image/mvtec
:link-type: doc

MVTec AD dataset datamodule for unsupervised anomaly detection.
:::

:::{grid-item-card} Visa
:link: image/visa
:link-type: doc

Visual Anomaly (VisA) dataset datamodule.
:::
```

## Video Datamodules

```{grid} 3
:gutter: 2

:::{grid-item-card} Avenue
:link: video/avenue
:link-type: doc

CUHK Avenue dataset datamodule for video anomaly detection.
:::

:::{grid-item-card} ShanghaiTech
:link: video/shanghaitech
:link-type: doc

ShanghaiTech dataset datamodule for video anomaly detection.
:::

:::{grid-item-card} UCSDped
:link: video/ucsdped
:link-type: doc

UCSD Pedestrian dataset datamodule for video anomaly detection.
:::
```

```{toctree}
:hidden:
:maxdepth: 1

depth/index
image/index
video/index
```

## Depth Datamodules

```{grid} 2
:gutter: 2

:::{grid-item-card} MVTec 3D
:link: depth/mvtec_3d
:link-type: doc

MVTec 3D-AD dataset datamodule for unsupervised 3D anomaly detection and localization.
:::

:::{grid-item-card} Folder 3D
:link: depth/folder_3d
:link-type: doc

Custom folder-based 3D datamodule for organizing your own depth-based anomaly detection dataset.
:::
```
