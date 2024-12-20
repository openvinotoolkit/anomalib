# Data Classes

Anomalib's dataclasses provide type-safe data containers with automatic validation. They support both PyTorch and NumPy backends for flexible data handling.

::::{grid} 1 2 2 3
:gutter: 3
:padding: 2
:class-container: landing-grid

:::{grid-item-card} {octicon}`package` Generic Classes
:link: generic
:link-type: doc
:class-card: custom-card

Base dataclasses that define common data structures and validation logic:

- Generic Item/Batch
- Input/Output Fields
- Validation Mixins

+++
[Learn More »](generic)
:::

:::{grid-item-card} {octicon}`cpu` PyTorch Classes
:link: torch
:link-type: doc
:class-card: custom-card

PyTorch tensor-based implementations:

- Image, Video, Depth Items
- Batch Processing Support
- Type-safe Validation

+++
[Learn More »](torch)
:::

:::{grid-item-card} {octicon}`database` NumPy Classes
:link: numpy
:link-type: doc
:class-card: custom-card

NumPy array-based implementations:

- Efficient Data Processing
- Array-based Containers
- Conversion Utilities

+++
[Learn More »](numpy)
:::
::::

## Documentation

For detailed documentation and examples, see:

- {doc}`Generic Base Classes <generic>`
- {doc}`PyTorch Classes <torch>`
- {doc}`NumPy Classes <numpy>`

```{toctree}
:hidden:

generic
torch
numpy
```
