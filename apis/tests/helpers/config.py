"""
Config Helpers for OTE Training
"""
from ote_sdk.configuration.configurable_parameters import ConfigurableParameters
from ote_sdk.configuration.helper import create
from ote_sdk.entities.model_template import ModelTemplate, parse_model_template


def get_config(template_file_path: str):

    model_template: ModelTemplate = parse_model_template(template_file_path)
    hyper_parameters: dict = model_template.hyper_parameters.data
    config: ConfigurableParameters = create(hyper_parameters)
    return config
