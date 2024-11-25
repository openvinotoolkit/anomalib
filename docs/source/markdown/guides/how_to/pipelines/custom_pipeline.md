# Pipelines

This guide demonstrates how to create a [Pipeline](../../reference/pipelines/index.md) for your custom task.

A pipeline is made up of runners. These runners are responsible for running a single type of job. A job is the smallest unit of work that is independent, such as, training a model or statistical comparison of the outputs of two models. Each job should be designed to be independent of other jobs so that they are agnostic to the runner that is running them. This ensures that the job can be run in parallel or serially without any changes to the job itself. The runner does not directly instantiate a job but rather has a job generator that generates the job based on the configuration. This generator is responsible for parsing the config and generating the job.

## Birds Eye View

In this guide we are going to create a dummy significant parameter search pipeline. The pipeline will have two jobs. The first job trains a model and computes the metric. The second job computes the significance of the parameters to the final score using shapely values. The final output of the pipeline is a plot that shows the contribution of each parameter to the final score. This will help teach you how to create a pipeline, a job, a job generator, and how to expose it to the `anomalib` CLI. The pipeline is going to be named `experiment`. So by the end of this you will be able to generate significance plot using

```{literalinclude} ../../../../snippets/pipelines/dummy/anomalib_cli.txt
:language: bash
```

The final directory structure will look as follows:

```{literalinclude} ../../../../snippets/pipelines/dummy/src_dir_structure.txt

```

```{literalinclude} ../../../../snippets/pipelines/dummy/tools_dir_structure.txt
:language: bash
```

## Creating the Jobs

Let's first look at the base class for the [jobs](../../reference/pipelines/base/job.md). It has a few methods defined.

- The `run` method is the main method that is called by the runner. This is where we will train the model and return the model metrics.
- The `collect` method is used to gather the results from all the runs and collate them. This is handy as we want to pass a single object to the next job that contains details of all the runs including the final score.
- The `save` method is used to write any artifacts to the disk. It accepts the gathered results as a parameter. This is useful in a variety of situations. Say, when we want to write the results in a csv file or write the raw anomaly maps for further processing.

Let's create the first job that trains the model and computes the metric. Since it is a dummy example, we will just return a random number as the metric.

```python
class TrainJob(Job):
    name = "train"

    def __init__(self, lr: float, backbone: str, stride: int):
        self.lr = lr
        self.backbone = backbone
        self.stride = stride

    def run(self, task_id: int | None = None) -> dict:
        print(f"Training with lr: {self.lr}, backbone: {self.backbone}, stride: {self.stride}")
        time.sleep(2)
        score = np.random.uniform(0.7, 0.1)
        return {"lr": self.lr, "backbone": self.backbone, "stride": self.stride, "score": score}
```

Ignore the `task_id` for now. It is used for parallel jobs. We will come back to it later.

````{note}
The `name` attribute is important and is used to identify the arguments in the job config file.
So, in our case the config `yaml` file will contain an entry like this:

```yaml
...
train:
    lr:
    backbone:
    stride:
...
````

Of course, it is up to us to choose what parameters should be shown under the `train` key.

Let's also add the `collect` method so that we return a nice dict object that can be used by the next job.

```python
def collect(results: list[dict]) -> dict:
        output: dict = {}
        for key in results[0]:
            output[key] = []
        for result in results:
            for key, value in result.items():
                output[key].append(value)
        return output
```

We can also define a `save` method that writes the dictionary as a csv file.

```python
@staticmethod
def save(results: dict) -> None:
    """Save results in a csv file."""
    results_df = pd.DataFrame(results)
    file_path = Path("runs") / TrainJob.name
    file_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(file_path / "results.csv", index=False)
```

The entire job class is shown below.

```{literalinclude} ../../../../snippets/pipelines/dummy/train_job.txt
:language: python
```

Now we need a way to generate this job when the pipeline is run. To do this we need to subclass the [JobGenerator](../../reference/pipelines/base/generator.md) class.

The job generator is the actual object that is attached to a runner and is responsible for parsing the configuration and generating jobs. It has two methods that need to be implemented.

- `generate_job`: This method accepts the configuration as a dictionary and, optionally, the results of the previous job. For the train job, we don't need results for previous jobs, so we will ignore it.
- `job_class`: This holds the reference to the class of the job that the generator will yield. It is used to inform the runner about the job that is being run, and is used to access the static attributes of the job such as its name, collect method, etc.

Let's first start by defining the configuration that the generator will accept. The train job requires three parameters: `lr`, `backbone`, and `stride`. We will also add another parameter that defines the number of experiments we want to run. One way to define it would be as follows:

```yaml
train:
  experiments: 10
  lr: [0.1, 0.99]
  backbone:
    - resnet18
    - wide_resnet50
  stride:
    - 3
    - 5
```

For this example the specification is defined as follows.

1. The number of experiments is set to 10.
2. Learning rate is sampled from a uniform distribution in the range `[0.1, 0.99]`.
3. The backbone is chosen from the list `["resnet18", "wide_resnet50"]`.
4. The stride is chosen from the list `[3, 5]`.

```{note}
While the `[ ]` and `-` syntax in `yaml` both signify a list, for visual disambiguation this example uses `[ ]` to denote closed interval and `-` for a list of options.
```

With this defined, we can define the generator class as follows.

```{literalinclude} ../../../../snippets/pipelines/dummy/train_generator.txt
:language: python
```

Since this is a dummy example, we generate the next experiment randomly. In practice, you would use a more sophisticated method that relies on your validation metrics to generate the next experiment.

```{admonition} Challenge
:class: tip
For a challenge define your own configuration and a generator to parse that configuration.
```

Okay, so now we can train the model. We still need a way to find out which parameters contribute the most to the final score. We will do this by computing the shapely values to find out the contribution of each parameter to the final score.

Let's first start by adding the library to our environment

```bash
pip install shap
```

The following listing shows the job that computes the shapely values and saves a plot that shows the contribution of each parameter to the final score. A quick rundown without going into the details of the job (as it is irrelevant to the pipeline) is as follows. We create a `RandomForestRegressor` that is trained on the parameters to predict the final score. We then compute the shapely values to identify the parameters that have the most significant impact on the model performance. Finally, the `save` method saves the plot so we can visually inspect the results.

```{literalinclude} ../../../../snippets/pipelines/dummy/significance_job.txt

```

Great! Now we have the job, as before, we need the generator. Since we only need the results from the previous stage, we don't need to define the config. Let's quickly write that as well.

```{literalinclude} ../../../../snippets/pipelines/dummy/significance_job_generator.txt

```

## Experiment Pipeline

So now we have the jobs, and a way to generate them. Let's look at how we can chain them together to achieve what we want. We will use the [Pipeline](../../reference/pipelines/base/pipeline.md) class to define the pipeline.

When creating a custom pipeline, there is only one important method that we need to implement. That is the `_setup_runners` method. This is where we chain the runners together.

```{literalinclude} ../../../../snippets/pipelines/dummy/pipeline_serial.txt
:language: python
```

In this example we use `SerialRunner` for running each job. It is a simple runner that runs the jobs in a serial manner. For more information on `SerialRunner` look [here](../../reference/pipelines/runners/serial.md).

Okay, so we have the pipeline. How do we run it? To do this let's create a simple entrypoint in `tools` folder of Anomalib.

Here is how the directory looks.

```{literalinclude} ../../../../snippets/pipelines/dummy/tools_dir_structure.txt
:language: bash
```

As you can see, we have the `config.yaml` file in the same directory. Let's quickly populate `experiment.py`.

```python
from anomalib.pipelines.experiment_pipeline import ExperimentPipeline

if __name__ == "__main__":
    ExperimentPipeline().run()
```

Alright! Time to take it on the road.

```bash
python tools/experimental/experiment/experiment.py --config tools/experimental/experiment/config.yaml
```

If all goes well you should see the summary plot in `runs/significant_feature/summary_plot.png`.

## Exposing to the CLI

Now that you have your shiny new pipeline, you can expose it as a subcommand to `anomalib` by adding an entry to the pipeline registry in `anomalib/cli/pipelines.py`.

```python
if try_import("anomalib.pipelines"):
    ...
    from anomalib.pipelines import ExperimentPipeline

PIPELINE_REGISTRY: dict[str, type[Pipeline]] | None = {
    "experiment": ExperimentPipeline,
    ...
}
```

With this you can now call

```{literalinclude} ../../../../snippets/pipelines/dummy/anomalib_cli.txt
:language: bash
```

Congratulations! You have successfully created a pipeline that trains a model and computes the significance of the parameters to the final score 🎉

```{admonition} Challenge
:class: tip
This example used a random model hence the scores were meaningless. Try to implement a real model and compute the scores. Look into which parameters lead to the most significant contribution to your score.
```

## Final Tweaks

Before we end, let's look at a few final tweaks that you can make to the pipeline.

First, let's run the initial model training in parallel. Since all jobs are independent, we can use the [ParallelRunner](../../reference/pipelines/runners/parallel.md). Since the `TrainJob` is a dummy job in this example, the pool of parallel jobs is set to the number of experiments.

```{literalinclude} ../../../../snippets/pipelines/dummy/pipeline_parallel.txt

```

You now notice that the entire pipeline takes lesser time to run. This is handy when you have large number of experiments, and when each job takes substantial time to run.

Now on to the second one. When running the pipeline we don't want our terminal cluttered with the outputs from each run. Anomalib provides a handy decorator that temporarily hides the output of a function. It suppresses all outputs to the standard out and the standard error unless an exception is raised. Let's add this to the `TrainJob`

```python
from anomalib.utils.logging import hide_output

class TrainJob(Job):
    ...

    @hide_output
    def run(self, task_id: int | None = None) -> dict:
        ...
```

You will no longer see the output of the `print` statement in the `TrainJob` method in the terminal.
