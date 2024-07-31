# Pipelines

```{danger}
The pipelines feature is experimental and might be changed without backward compatibility.
```

## Introduction

Tasks such as Benchmarking, Ensemble Tiling, and Hyper-parameter optimization requires running multiple models and chaining multiple stages together. The pipelines feature provides a way to define and run such tasks. Each part of the pipeline is designed to be independent and composable so that they can be reused across different pipelines.

## Terminology

- **Pipeline**: Pipeline is the main entity that defines the sequence of [jobs](./base/job.md) to be executed. It is responsible for creating and running the jobs. The job itself is generated using a [job generator](./base/generator.md). And, these are chained using a [runner](./runners/index.md).

- **Runner**: A runner is responsible for scheduling and running the jobs. It also passes the output of the previous job, if available. It also calls the right hooks to gather and save the results from the jobs and passes the gathered results to the next runner.

- **Job Generator**: The job generator is responsible for generating jobs based on the configuration. It is used by the runner to create jobs.

- **Job**: A job is an atomic unit of work that can be run independently. It is responsible for running a single task. For example, training a model or computing metrics. The idea behind this is to ensure that it can be attached to any runner without making changes to the job itself. This is useful when you want to distribute the jobs to increase the throughput of your pipeline.

```{admonition} Detailed Walkthrough
:class: tip
For more clarity on creating a custom pipeline, refer to the [How-To Guide](../../how_to/pipelines/index.md).
```

## Base classes

::::{grid}
:margin: 1 1 0 0
:gutter: 1

:::{grid-item-card} {octicon}`workflow` Pipeline
:link: ./base/pipeline
:link-type: doc

Base class for pipeline.
:::

:::{grid-item-card} {octicon}`file` Job
:link: ./base/job
:link-type: doc

Base class for job.
:::

:::{grid-item-card} {octicon}`iterations` Job Generator
:link: ./base/generator
:link-type: doc

Base class for job generator.
:::

:::{grid-item-card} {octicon}`play` Runner
:link: ./runners/index
:link-type: doc

Base class for runner.
:::

::::

## Available Pipelines

::::{grid}
:margin: 1 1 0 0
:gutter: 1

:::{grid-item-card} {octicon}`number` Benchmarking
:link: ./benchmark/index
:link-type: doc

Compute metrics across models using a grid-search.
:::

::::

```{toctree}
:caption: Pipelines
:hidden:

./benchmark/index
./runners/index
```
