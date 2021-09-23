from unittest import mock

import numpy as np
import pytest

from anomalib.loggers.sigopt import SigoptLogger


@mock.patch("anomalib.loggers.sigopt.sigopt")
def test_sigopt_logger_init(sigopt):
    logger = SigoptLogger(name="test_name", project="test_project")
    logger.log_metrics({"acc": 1.0})
    sigopt.create_run.assert_called_once_with(name="test_name", project="test_project")
    sigopt.create_run().log_metric.assert_called_once_with(name="acc", value=1.0)
    sigopt.create_run().log_checkpoint.assert_called_once_with({"acc": 1.0})


@mock.patch("anomalib.loggers.sigopt.sigopt")
def test_sigopt_logger_experiment_object(sigopt):
    connection = sigopt.Connection()

    experiment = connection.experiments().create(
        name="test_name",
        project="test_project",
        observation_budget=10,
        parameters=[
            dict(name="test_param_1", type=int, bounds=dict(min=1, max=10)),
            dict(name="test_param_2", type=float, bounds=dict(min=0.1, max=0.9)),
        ],
        metrics=[dict(name="test_metric", objective="maximize")],
    )

    logger = SigoptLogger(name="test_name", project="test_project", experiment=experiment)
    logger.log_metrics({"acc": 1.0})
    sigopt.Connection().experiments().create().log_metric.assert_called_once_with(name="acc", value=1.0)
    sigopt.Connection().experiments().create().log_checkpoint.assert_called_once_with({"acc": 1.0})


@mock.patch("anomalib.loggers.sigopt.sigopt")
def test_sigopt_logger_frequency(sigopt):

    # should log every two steps
    logger = SigoptLogger(name="test_name", project="test_project", max_epochs=400)
    logger.log_checkpoint({"acc": 1.0}, epoch=0)
    sigopt.create_run().log_checkpoint.assert_not_called()
    logger.log_checkpoint({"acc": 1.0}, epoch=1)
    sigopt.create_run().log_checkpoint.assert_called_once()


def type_checker(param: dict):
    for k, v in param.items():
        # uses type to check as we don't want to check the subclasses
        if type(v) != int and type(v) != float and type(v) != str:
            raise TypeError(f"Dict contains unsupported types {k}:{v} - {type(v)}")


@mock.patch("anomalib.loggers.sigopt.sigopt")
def test_sigopt_logger_hyperparameter(sigopt):
    logger = SigoptLogger(name="test_name", project="test_project")
    hparams = {
        "lr": 1e-3,
        "layers": ["layer_1", "layer_2"],
        "i_lst": [1, 2, 3],
        "f_lst": [0.1, 3.0, 0.5],
        "dict": {"a": 1, "b": 2, "c": 3},
        "str": "test_val",
        2: "test_val_2",
        "bool1": False,
        "bool2": True,
        True: False,
    }

    # test basic logging function
    logger = SigoptLogger(name="test_name", project="test_project", max_epochs=400)
    logger.log_hyperparams(hparams)

    def fake_set_params(_, param):
        param = SigoptLogger._convert_params(param)
        param = SigoptLogger._flatten_dict(param)
        param = SigoptLogger._sanitize_callable_params(param)
        param = SigoptLogger._sanitize_other_params(param)
        return param

    # passes when parameters have been properly sanitized
    with mock.patch("anomalib.loggers.sigopt.SigoptLogger.log_hyperparams", fake_set_params):
        ret = logger.log_hyperparams(hparams)
        assert type(ret) == dict
        # raises type error if the dict has not been properly sanitized.
        type_checker(ret)

    def fake_sanitize_other_params(_, param):
        param = SigoptLogger._convert_params(param)
        param = SigoptLogger._flatten_dict(param)
        param = SigoptLogger._sanitize_callable_params(param)
        return param

    # passes when parameters have not been properly sanitized and leads to an exception being thrown.
    with mock.patch("anomalib.loggers.sigopt.SigoptLogger.log_hyperparams", fake_sanitize_other_params):
        ret = logger.log_hyperparams(hparams)
        # passes when error is raised
        with pytest.raises(TypeError):
            ret = logger.log_hyperparams(hparams)
            type_checker(ret)


@mock.patch("anomalib.loggers.sigopt.sigopt")
def test_sigopt_add_images(sigopt):
    logger = SigoptLogger(name="test_name", project="test_project")
    # without name
    logger.add_image(image=np.ones((255, 255, 3)))
    # with name
    logger.add_image(image=np.ones((255, 255, 3)), name="test_image")
