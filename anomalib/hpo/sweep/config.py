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


def get_parameters(configs: Dict) -> List[Dict]:
    """Return list of params

    Args:
      configs: Dict: Parameter configurations. Assumes that these parameters have been sanitized and validated.
    Returns:
      list of dict containing parameters in sigopt format

    """
    params = []
    for config in configs.items():
        # TODO add support for categorical values. https://jira.devtools.intel.com/browse/IAAALD-35
        param = None
        if "grid" in config[1]:
            param = dict(name=config[0], type=config[1].type, grid=config[1].grid)
        elif config[1].type in ["double", "int"]:
            param = dict(
                name=config[0], type=config[1].type, bounds=dict(min=float(config[1].min), max=float(config[1].max))
            )
        if param is not None:
            params.append(param)

    return params


def validate_search_params(params: Dict[str, str]):
    """Validates the keys and data types in the parameters section of hpo"""

    keys = set(params.keys())  # cache keys in set for faster lookup

    # Check if grid or range is passed

    ## check if the right key exists. There might be a better way to do this
    if len(set(["grid", "min", "max"]).intersection(keys)) == 0:
        raise IncorrectHPOConfiguration("Expected search parameters to have either grid or a range(min, max)")

    ## check if only one of grid/range is passed
    if ("grid" in keys and (len(set(["min", "max"]).intersection(keys)) != 0)) or (
            (len(set(["min", "max"]).intersection(keys)) == 2) and "grid" in keys
    ):
        raise IncorrectHPOConfiguration("Found both grid and range(min,max) keys in configuration. Please use only one")

    ## Check datatype
    if "grid" in keys:
        if params["type"] == "double":
            for val in params["grid"]:
                if not isinstance(val, float):
                    raise IncorrectHPOConfiguration(
                        f"Type mismatch in parameter configuration. Expected float. Found {type(val)}"
                    )
        elif params["type"] == "int":
            for val in params["grid"]:
                if not isinstance(val, int):
                    raise IncorrectHPOConfiguration(
                        f"Type mismatch in parameter configuration. Expected Integer. Found {type(val)}"
                    )
    else:
        if params["type"] == "double" and (
                not isinstance(params["min"], float) or not isinstance(params["max"], float)
        ):
            raise IncorrectHPOConfiguration(
                f"Type mismatch in parameter configuration. Expected float."
                f"Found min: {type(params['min'])}, max: {type(params['max'])}"
            )
        if params["type"] == "int" and (not isinstance(params["min"], int) or not isinstance(params["max"], int)):
            raise IncorrectHPOConfiguration(
                f"Type mismatch in parameter configuration. Expected Integer."
                f"Found min: {type(params['min'])}, max: {type(params['max'])}"
            )


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

    if config.hyperparameter_search.metric.objective not in ["maximize", "minimize"]:
        raise IncorrectHPOConfiguration("Objective should be one of [maximize, minimize]")

    # test if the ranges are valid (datatype corresponds to range values)
    for param in config.hyperparameter_search.parameters.values():
        validate_search_params(params=param)


def get_experiment(connection: Connection, config: Union[DictConfig, ListConfig]) -> Experiment:
    """Returns the sigopt experiment object

    Args:
      connection: Connection: Connection Object
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
