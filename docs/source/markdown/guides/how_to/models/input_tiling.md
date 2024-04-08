# Input tiling

This tutorial will show you how to tile the input to a model, using the {py:class}`Tiler <anomalib.data.utils.tiler.Tiler>`.

```{warning}
This tutorial assumes that you have already installed anomalib.
If not, please refer to the [Installation](../../../../index.md#installation) section.
```

```{warning}
Only selected models support tiling.
In the current version of Anomalib, these are:

- [Padim](../../reference/models/image/padim.md)
- [Patchcore](../../reference/models/image/patchcore.md)
- [Reverse Distillation](../../reference/models/image/reverse_distillation.md)
- [STFPM](../../reference/models/image/stfpm.md)

```

## General tiling information

The general idea of input tiling is that the image is split into a rectangular grid of tiles as a pre-processing step, usually in order to reduce memory usage.
By passing individual tiles to the model as input instead of full images, tiling reduces the model's input dimensions, while maintaining the effective input resolution of the images content-wise.

```{note}
Tiler in Anomalib by default stacks the tiles batch-wise, so the memory consumption stays unchanged if the batch size is not reduced.
```

The process of tiling is parametrized by four parameters `tile_size`, `stride`, `remove_border_count`, and `mode`.

- `tile_size` - determines the size of our tiles. Can be either a single number (square tiles) or a tuple.
- `stride` - determines by how much we move in each direction when "cutting" the image into tiles. Can be either a single number (same step in both directions) or a tuple.
- `remove_border_count` - how many pixels are removed at the border of the image before tiling (defaults to 0).
- `mode` - what type of upscaling is used when the image isn't exactly divisible into tile-set specified by the parameters `tile_size` and `stride` (defaults to padding).

In most cases, we are only interested in the first two parameters - `tile_size` and `stride`. For the other two, refer to [Tiler implementation](../../reference/data/utils/tiling.md).

## Tiling setup

We can utilize the tiling in two ways. Either with the CLI or by using the API.
In both cases, we need to use the {py:class}`TilerConfigurationCallback <anomalib.callbacks.TilerConfigurationCallback>`.
This callback is responsible for assigning the tiler object to the model before the training starts.

```{note}
Besides the arguments from {py:class}`Tiler <anomalib.data.utils.tiler.Tiler>`, {py:class}`TilerConfigurationCallback <anomalib.callbacks.TilerConfigurationCallback>` also has an additional `enable` argument, which must be set to `True` if we want the tiling to happen.
```

::::{tab-set}

:::{tab-item} API

To use tiling from the API, we need to initialize the {py:class}`TilerConfigurationCallback <anomalib.callbacks.TilerConfigurationCallback>` and pass it to the engine:

```{code-block} python
:lineno-start: 1
:emphasize-lines: 12, 15
# Import the required modules
from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import Padim
from anomalib.callbacks import TilerConfigurationCallback

# Initialize the datamodule and model
datamodule = MVTec(num_workers=0, image_size=(128, 128))
model = Padim()

# prepare tiling configuration callback
tiler_config_callback = TilerConfigurationCallback(enable=True, tile_size=[128, 64], stride=64)

# pass the tiling configuration callback to the engine
engine = Engine(image_metrics=["AUROC"], pixel_metrics=["AUROC"], callbacks=[tiler_config_callback])

# train the model (tiling is seamlessly utilized in the background)
engine.fit(datamodule=datamodule, model=model)
```

:::

:::{tab-item} CLI

### Using CLI arguments

We can set the {py:class}`TilerConfigurationCallback <anomalib.callbacks.TilerConfigurationCallback>` and its init arguments directly from the CLI.

We pass it as trainer.callback, and then provide the parameters:

```{code-block} bash
:emphasize-lines: 2, 3, 4, 5
anomalib train --model Padim --data anomalib.data.MVTec
    --trainer.callbacks anomalib.callbacks.tiler_configuration.TilerConfigurationCallback
    --trainer.callbacks.enable True
    --trainer.callbacks.tile_size 128
    --trainer.callbacks.stride 64
```

### Using config

For more advanced configuration, we can prepare the config file:

```{code-block} yaml
:lineno-start: 1
trainer.callbacks:
  class_path: anomalib.callbacks.tiler_configuration.TilerConfigurationCallback
  init_args:
    enable: True
    tile_size: [128, 256]
    stride: 64
```

Then use the config from the CLI:

```{code-block} bash
anomalib train --model Padim --data anomalib.data.MVTec --config config.yaml
```

:::

::::
