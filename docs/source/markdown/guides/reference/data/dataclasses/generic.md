# Generic Dataclasses

The generic dataclasses module provides the foundational data structures and validation logic used throughout Anomalib. These classes are designed to be flexible and type-safe, serving as the base for both PyTorch and NumPy implementations.

```{eval-rst}
.. currentmodule:: anomalib.data.dataclasses.generic
```

## Core Concepts

### Type Variables

The module uses several type variables to ensure type safety across different implementations:

- `ImageT`: Type variable for image data (PyTorch Image/Video or NumPy array)
- `T`: Type variable for tensor-like data (PyTorch Tensor or NumPy array)
- `MaskT`: Type variable for mask data (PyTorch Mask or NumPy array)
- `PathT`: Type variable for path data (string or list of strings)

## Base Classes

### InputFields

```{eval-rst}
.. autoclass:: _InputFields
   :members:
   :show-inheritance:
```

### ImageInputFields

```{eval-rst}
.. autoclass:: _ImageInputFields
   :members:
   :show-inheritance:
```

### VideoInputFields

```{eval-rst}
.. autoclass:: _VideoInputFields
   :members:
   :show-inheritance:
```

### DepthInputFields

```{eval-rst}
.. autoclass:: _DepthInputFields
   :members:
   :show-inheritance:
```

### OutputFields

```{eval-rst}
.. autoclass:: _OutputFields
   :members:
   :show-inheritance:
```

## Mixins

### UpdateMixin

```{eval-rst}
.. autoclass:: UpdateMixin
   :members:
   :show-inheritance:
```

### BatchIterateMixin

```{eval-rst}
.. autoclass:: BatchIterateMixin
   :members:
   :show-inheritance:
```

## Generic Classes

### GenericItem

```{eval-rst}
.. autoclass:: _GenericItem
   :members:
   :show-inheritance:
```

### GenericBatch

```{eval-rst}
.. autoclass:: _GenericBatch
   :members:
   :show-inheritance:
```

## Field Validation

### FieldDescriptor

```{eval-rst}
.. autoclass:: FieldDescriptor
   :members:
   :show-inheritance:
```

## See Also

- {doc}`torch`
- {doc}`numpy`
