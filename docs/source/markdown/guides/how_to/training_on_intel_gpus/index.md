# Training on Intel GPUs

This tutorial demonstrates how to train a model on Intel GPUs using anomalib.
Anomalib comes with XPU accelerator and strategy for PyTorch Lightning. This allows you to train your models on Intel GPUs.

> [!Note]
> Currently, only single GPU training is supported on Intel GPUs.
> These commands were tested on Arc 750 and Arc 770.

## Installing Drivers

First, check if you have the correct drivers installed. If you are on Ubuntu, you can refer to the [following guide](https://dgpu-docs.intel.com/driver/client/overview.html).

Another recommended tool is `xpu-smi` which can be installed from the [releases](https://github.com/intel/xpumanager) page.

If everything is installed correctly, you should be able to see your card using the following command:

```bash
xpu-smi discovery
```

## Installing PyTorch

Then, ensure that you have PyTorch with XPU support installed. For more information, please refer to the [PyTorch XPU documentation](https://pytorch.org/docs/stable/notes/get_start_xpu.html)

To ensure that your PyTorch installation supports XPU, you can run the following command:

```bash
python -c "import torch; print(torch.xpu.is_available())"
```

If the command returns `True`, then your PyTorch installation supports XPU.

## 🔌 API

```python
from anomalib.data import MVTecAD
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import Stfpm

engine = Engine(
    strategy=SingleXPUStrategy(),
    accelerator=XPUAccelerator(),
)
engine.train(Stfpm(), datamodule=MVTecAD())
```

## ⌨️ CLI

```bash
anomalib train --model Padim --data MVTecAD --trainer.accelerator xpu --trainer.strategy xpu_single
```
