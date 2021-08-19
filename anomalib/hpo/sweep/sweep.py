"""Loop for running the hpo sweep"""

from typing import Union

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from pytorch_lightning import Trainer
from sigopt import Connection

from anomalib.core.callbacks.visualizer_callback import VisualizerCallback
from anomalib.datasets import get_datamodule
from anomalib.hpo.sweep.config import get_experiment
from anomalib.models import get_model


def run_sweep(config: Union[DictConfig, ListConfig]) -> None:
    """
    Encapsulates the hpo sweep loop

    Args:
        config: configs loaded with omegaconf

    Returns: None

    """

    connection = Connection()
    experiment = get_experiment(connection=connection, config=config)
    print("Created: https://app.sigopt.com/experiment/" + experiment.id)

    while experiment.progress.observation_count < experiment.observation_budget:

        datamodule = get_datamodule(config)

        suggestion = connection.experiments(experiment.id).suggestions().create()

        # get names of the suggested params
        params = suggestion.assignments.keys()
        # replace the params in config with suggested param values
        for param in params:
            config.model[param] = suggestion.assignments[param]

        model = get_model(config)

        # remove visualizer_callback if it is in model
        for index, callback in enumerate(model.callbacks):
            if isinstance(callback, VisualizerCallback):
                model.callbacks.pop(index)

        trainer = Trainer(callbacks=model.callbacks, **config.trainer, logger=False)
        trainer.fit(model=model, datamodule=datamodule)
        result = trainer.test(model=model, datamodule=datamodule)

        # return at least 1 metric
        if len(result[0]) == 0:
            raise ValueError("Test step returned no metrics")

        # get the value of the required metric from the results
        for metrics in result:
            if experiment.metric.name in metrics.keys():
                value = metrics[experiment.metric.name]

        # is not assigned anything. Raises key error as key is not in dict.
        if "value" not in locals():
            raise KeyError(
                f"Model does not return {experiment.metric.name} from test step. It is also possible that"
                f"you might have `prog_bar=True` in self.log()"
            )

        connection.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=value,
        )

        # upload experiment result
        connection.experiments(experiment.id).observations().create(suggestion=suggestion.id, value=value)
        experiment = connection.experiments(experiment.id).fetch()
