# Depth Datamodules

Anomalib provides datamodules for handling depth-based anomaly detection datasets. These datamodules are designed to work with both RGB and depth information for 3D anomaly detection tasks.

## Available Datamodules

```{grid} 2
:gutter: 2

:::{grid-item-card} MVTec 3D
:link: mvtec_3d
:link-type: doc

MVTec 3D-AD dataset datamodule for unsupervised 3D anomaly detection and localization.
:::

:::{grid-item-card} Folder 3D
:link: folder_3d
:link-type: doc

Custom folder-based 3D datamodule for organizing your own depth-based anomaly detection dataset.
:::
```

```{toctree}
:hidden:
:maxdepth: 1

mvtec_3d
folder_3d
```
