"""Loop for running the hpo sweep"""
import copy
import operator
from functools import reduce
from typing import Any, List, Union

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from pytorch_lightning import Trainer
from sigopt import Connection

from anomalib.config import update_input_size_config
from anomalib.core.callbacks.visualizer_callback import VisualizerCallback
from anomalib.datasets import get_datamodule
from anomalib.hpo.sweep.config import get_experiment
from anomalib.models import get_model


def get_from_nested_config(config: Union[DictConfig, ListConfig], keymap: List) -> Any:
    """
    Retrieves an item from a nested config object using a list of keys.

    Args:
        config: DictConfig: nested DictConfig object
        keymap: List[str]: list of keys corresponding to item that should be retrieved.
    """
    return reduce(operator.getitem, keymap, config)


def set_in_nested_config(config: Union[DictConfig, ListConfig], keymap: List, value: Any):
    """
    Set an item in a nested config object using a list of keys.

    Args:
        config: DictConfig: nested DictConfig object
        keymap: List[str]: list of keys corresponding to item that should be set.
        value: Any: Value that should be assigned to the dictionary item at the specified location.
    """
    get_from_nested_config(config, keymap[:-1])[keymap[-1]] = value


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

    base_config = copy.deepcopy(config)

    while experiment.progress.observation_count < experiment.observation_budget:
        # reset config
        config = copy.deepcopy(base_config)

        suggestion = connection.experiments(experiment.id).suggestions().create()

        # get names of the suggested params
        params = suggestion.assignments.keys()
        # replace the params in config with suggested param values
        for param in params:
            set_in_nested_config(config, param.split("."), suggestion.assignments[param])

        config = update_input_size_config(config)

        model = get_model(config)
        datamodule = get_datamodule(config)

        # remove visualizer_callback if it is in model
        for index, callback in enumerate(model.callbacks):
            if isinstance(callback, VisualizerCallback):
                model.callbacks.pop(index)
                break

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

        # upload experiment result
        connection.experiments(experiment.id).observations().create(suggestion=suggestion.id, value=value)
        experiment = connection.experiments(experiment.id).fetch()
