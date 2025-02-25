# Data

A comprehensive data handling pipeline with modular components for anomaly detection tasks.

::::{grid} 1 2 2 3
:gutter: 3
:padding: 2
:class-container: landing-grid

:::{grid-item-card} {octicon}`package` Data Classes
:link: ./dataclasses/index
:link-type: doc
:class-card: custom-card

Core data structures that define how data is represented and validated throughout the pipeline. Features type-safe containers, dual backend support, and automatic validation.

+++
[Learn more »](./dataclasses/index)
:::

:::{grid-item-card} {octicon}`database` Datasets
:link: ./datasets/index
:link-type: doc
:class-card: custom-card

Ready-to-use PyTorch Dataset implementations of standard benchmark datasets (MVTecAD, BTech) and support for custom datasets across multiple modalities (Image, Video, Depth).

+++
[Learn more »](./datasets/index)
:::

:::{grid-item-card} {octicon}`workflow` Data Modules
:link: ./datamodules/index
:link-type: doc
:class-card: custom-card

Lightning implementations of these PyTorch datasets that provide automated data loading, train/val/test splitting, and distributed training support through the PyTorch Lightning DataModule interface.

+++
[Learn more »](./datamodules/index)
:::
::::

## Additional Resources

::::{grid} 2 2 2 2
:gutter: 2
:padding: 1

:::{grid-item-card} {octicon}`tools` Data Utils
:link: ./utils/index
:link-type: doc

Helper functions and utilities for data processing and augmentation.
:::

:::{grid-item-card} {octicon}`book` Tutorials
:link: ../tutorials/index
:link-type: doc

Step-by-step guides on using the data components.
:::
::::

```{toctree}
:caption: Data Components
:hidden:

./dataclasses/index
./datasets/index
./datamodules/index
./utils/index
```
