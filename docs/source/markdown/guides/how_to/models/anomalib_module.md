# AnomalibModule

`AnomalibModule` is the base Lightning module used by Anomalib models. It
coordinates preprocessing, post-processing, training, validation, testing,
prediction, and model export.

Model implementations normally subclass `AnomalibModule` and provide their
model-specific feature extraction, loss, and anomaly-score logic.

## API Reference

```{eval-rst}
.. autoclass:: anomalib.models.components.base.anomalib_module.AnomalibModule
   :members:
   :show-inheritance:
```
