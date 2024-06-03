# Runner

```{eval-rst}
.. autoclass:: anomalib.pipelines.components.base.runner.Runner
    :members:

```

## Available Runners

Anomalib provides a few runners that can be used in your pipelines.

::::{grid}
:margin: 1 1 0 0
:gutter: 1

:::{grid-item-card} {octicon}`list-ordered` Serial Runner
:link: ./serial
:link-type: doc

Runner for serial jobs.
:::

:::{grid-item-card} {octicon}`git-branch` Parallel Runner
:link: ./parallel
:link-type: doc

Runner for parallel jobs.
:::

::::

```{toctree}
:caption: Runners
:hidden:

./serial
./parallel
```
