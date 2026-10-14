# SuperADD

Implementation of the SuperADD model for anomaly detection and segmentation.

## Architecture

```{eval-rst}
.. image:: ../../../../../images/super_add/architecture.png
    :alt: SuperADD Architecture
```

SuperADD extracts multi-layer Vision Transformer token features from a pretrained DINOv3 backbone over overlapping image patches, builds a per-layer memory bank from normal training images using distance-based coreset subsampling, and detects anomalies by nearest-neighbor search against this memory bank.

## Usage

CLI training with default configuration:

```bash
anomalib train --model SuperADD --data MVTecAD2 --data.category bottle
```

Python API:

```python
from anomalib.data import MVTecAD2
from anomalib.engine import Engine
from anomalib.models import SuperADD

datamodule = MVTecAD2()
model = SuperADD()
engine = Engine()

engine.fit(datamodule=datamodule, model=model)
```

## API Reference

```{eval-rst}
.. automodule:: anomalib.models.image.super_add
   :members:
   :show-inheritance:
```
