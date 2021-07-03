from argparse import ArgumentParser
from typing import Union
from pytorch_lightning import Trainer, seed_everything
from sigopt import Connection
from sigopt.resource import ApiResource
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from anomalib.config.config import get_configurable_parameters
from anomalib.config.sweep_config import get_experiment
from anomalib.datasets import get_datamodule
from anomalib.models import get_model



def sweep(connection: Connection, experiment: ApiResource , config: Union[DictConfig, ListConfig]) -> None:
    """
    Encapsulates the hpo sweep loop
    Args:
        connection: sigopt connection object
        experiment: sigopt experiment object
        config: configs loaded with omegaconf

    Returns: None

    """

    while experiment.progress.observation_count < experiment.observation_budget:

        datamodule = get_datamodule(config)

        suggestion = connection.experiments(experiment.id).suggestions().create()

        # get names of the suggested params
        params = suggestion.assignments.keys()
        # replace the params in config with suggested param values
        for param in params:
            config.model[param] = suggestion.assignments[param]

        model = get_model(config)

        trainer = Trainer(callbacks=model.callbacks, **config.trainer)
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
        if 'value' not in locals():
            raise KeyError(f"Model does not return {experiment.metric.name} from test step. It is also possible that"
                             f"you might have `prog_bar=True` in self.log()")

        connection.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=value,
        )

        # upload experiment result
        connection.experiments(experiment.id).observations().create(suggestion=suggestion.id, value=value)
        experiment = connection.experiments(experiment.id).fetch()


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="stfpm", help="Name of the algorithm to train/test")
    parser.add_argument("--model_config_path", type=str, required=False, help="Path to a model config file")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = get_configurable_parameters(model_name=args.model, model_config_path=args.model_config_path)

    if config.project.seed != 0:
        seed_everything(config.project.seed)

    connection = Connection()
    experiment = get_experiment(connection=connection, config=config)
    print("Created: https://app.sigopt.com/experiment/" + experiment.id)

    sweep(connection=connection, experiment=experiment, config=config)
