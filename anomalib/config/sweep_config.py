from typing import Dict, List

from omegaconf import OmegaConf
from sigopt import Connection
from sigopt.objects import Experiment


def _get_parameters(configs:Dict)-> List:
    """Return list of params

    :param configs: parameter configurations
    :return: list of params
    """
    params = []
    for config in configs.items():
        # TODO add support for categorical values
        if config[1].type in ['double', 'int']:
            param = dict(name=config[0], type=config[1].type, bounds=dict(min=float(config[1].min), max=float(config[1].max)))
            params.append(param)

    return params



def get_experiment(connection: Connection, sweep_config_path: str)->Experiment:
    config = OmegaConf.load(sweep_config_path)
    experiment = connection.experiments().create(
        name=config.name,
        project = config.project,
        observation_budget=config.observation_budget,
        parameters = _get_parameters(config.parameters),
        metrics = [dict(name=config.metric.name, objective=config.metric.objective)]
    )

    return experiment
