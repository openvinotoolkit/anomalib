# Benchmarking Pipeline

The benchmarking pipeline allows you to run multiple models across combination of parameters and dataset categories to collect metrics. The benchmarking run is configured using a config file that specifies the grid-search parameters. A sample config file is shown below:

```yaml
accelerator:
  - cuda
  - cpu
benchmark:
  seed: 42
  model:
    class_path:
      grid_search: [Padim, Patchcore]
  data:
    class_path: MVTec
    init_args:
      category:
        grid:
          - bottle
          - cable
          - capsule
```

The `accelerator` parameter is specific to the pipeline and is used to configure the runners. When `cuda` is passed it adds a [parallel](../runners/parallel.md) runner with number of jobs equal to the number of cuda devices. The idea is that since job is independent, we can increase the throughput by distributing each on an individual accelerator. The `cpu` jobs are run [serially](../runners/serial.md).

## Running the Benchmark Pipeline

There are two ways to run the benchmark pipeline; as a subcommand, or as a standalone entrypoint.

:::::{dropdown} CLI
:icon: code

::::{tab-set}
:::{tab-item} Anomalib subcommand
:sync: label-1

```{literalinclude} ../../../../../snippets/pipelines/benchmark/cli_anomalib.txt
:language: bash
```

:::

:::{tab-item} Standalone entrypoint
:sync: label-2

```{literalinclude} ../../../../../snippets/pipelines/benchmark/cli_tools.txt
:language: bash
```

:::

:::::

## Benchmark Pipeline Class

```{eval-rst}

.. autoclass:: anomalib.pipelines.benchmark.pipeline.Benchmark
    :members:
    :inherited-members:
    :show-inheritance:

```

::::{grid}
:margin: 1 1 0 0
:gutter: 1

:::{grid-item-card} Job
:link: ./job
:link-type: doc

Benchmark Job
:::

:::{grid-item-card} Generator
:link: ./generator
:link-type: doc

Benchmark Job Generator
:::

::::

```{toctree}
:caption: Benchmark
:hidden:

./job
./generator
```
