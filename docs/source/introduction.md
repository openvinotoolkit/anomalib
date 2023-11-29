# Introduction

::::{tab-set}

:::{tab-item} API

```{code-block} python
from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.models import Engine

datamodule = MVTec()
model = Patchcore()
engine = Engine()

engine.train(datamodule=datamodule, model=model)
```

:::

:::{tab-item} CLI
Content 2
:::

::::

:::{dropdown} Trainin
Dropdown content
:::
