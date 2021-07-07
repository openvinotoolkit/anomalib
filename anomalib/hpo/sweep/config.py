"""
Helper functions to get hpo experiment and read configurations
"""

from typing import Dict, List, Union

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from sigopt import Connection
from sigopt.objects import Experiment


class MissingHPOConfiguration(Exception):
    """Raised when configuration does not contain values for hyperparameter optimization"""


class IncorrectHPOConfiguration(Exception):
    """Raise when the HPO has incorrect exception"""


def get_parameters(configs: Dict) -> List:
    """Return list of params

    Args:
      configs: Dict: parameter configurations
    Returns:
      list of dict containing parameters in sigopt format

    """
    params = []
    for config in configs.items():
        # TODO add support for categorical values
        if config[1].type in ["double", "int"]:
            param = dict(
                name=config[0], type=config[1].type, bounds=dict(min=float(config[1].min), max=float(config[1].max))
            )
            params.append(param)

    return params


def validate_dtypes(dtype: str, min_val: Union[float, int], max_val: Union[float, int]):
    """Validates the dtypes in the parameters section of hpo"""
    if dtype == "double" and (not isinstance(min_val, float) or not isinstance(max_val, float)):
        raise IncorrectHPOConfiguration("Type mismatch in parameter configuration. Expected float")
    if dtype == "int" and (not isinstance(min_val, int) or not isinstance(max_val, int)):
        raise IncorrectHPOConfiguration("Type mismatch in parameter configuration. Expected integer")


def validate_config(config: Union[DictConfig, ListConfig]):
    """validates the passed configuration"""
    if "hyperparameter_search" not in config.keys():
        raise MissingHPOConfiguration(
            f"Config file for {config.model.name} does not contain parameters for hyperparameter optimization."
        )

    # test if only one metric is present (name, objective)
    if len(config.hyperparameter_search.metric) != 2:
        raise IncorrectHPOConfiguration("Optimization metric should use one metric. Please provide name and objective")

    for key in ["name", "objective"]:
        if key not in config.hyperparameter_search.metric.keys():
            raise KeyError(f"Missing key {key} in hpo metrics")

    if (
        "maximize" not in config.hyperparameter_search.metric.objective
        and "minimize" not in config.hyperparameter_search.metric.objective
    ):
        raise IncorrectHPOConfiguration("Objective should be one of [maximize, minimize]")

    # test if the ranges are valid (datatype corresponds to range values)
    for param in config.hyperparameter_search.parameters.values():
        validate_dtypes(dtype=param.type, min_val=param.min, max_val=param.max)


def get_experiment(connection: Connection, config: Union[DictConfig, ListConfig]) -> Experiment:
    """Returns the sigopt experiment object

    Args:
      connection: Connection:Connection Object
      config: Union[DictConfig |ListConfig]: Config read by omegaconf


    Returns: Experiment object created from the parameters
    """

    validate_config(config)
    experiment = connection.experiments().create(
        name=config.model.name,
        project=config.hyperparameter_search.project,
        observation_budget=config.hyperparameter_search.observation_budget,
        parameters=get_parameters(config.hyperparameter_search.parameters),
        metrics=[
            dict(name=config.hyperparameter_search.metric.name, objective=config.hyperparameter_search.metric.objective)
        ],
        metadata=dict(dataset=config.dataset.name),
    )

    return experiment
