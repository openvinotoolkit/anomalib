# Parallel Runner

The parallel runners creates a pool of runners with the pool size equal to the number defined when creating the runner.

Each process in the pool has a process id assigned to it that is between 0-`n_jobs`. When a job is run using the parallel runner, the process id is passed to the job. The job can use this id to determine process specific logic. For example, if the pool size is equal to the number of GPUs, the job can use the process id to assign a specific GPU to the process.

```{eval-rst}
.. automodule:: anomalib.pipelines.components.runners.parallel
   :members:
   :show-inheritance:
```
