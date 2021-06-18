from unittest import mock

import numpy as np
from anomalib.loggers.sigopt import SigoptLogger

# logging images
# TODO test for exception generated after logging more than 200 values


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
        metrics=[dict(name="test_metric", objective="maxamize")],
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
        1: "test_val_2",
    }

    def fake_set_params(_, param):
        param = SigoptLogger._convert_params(param)
        param = SigoptLogger._flatten_dict(param)
        param = SigoptLogger._sanitize_callable_params(param)
        param = SigoptLogger._sanitize_other_params(param)
        return param

    with mock.patch("anomalib.loggers.sigopt.SigoptLogger.log_hyperparams", fake_set_params):
        ret = logger.log_hyperparams(hparams)
        assert type(ret) == dict


@mock.patch("anomalib.loggers.sigopt.sigopt")
def test_sigopt_log_images(sigopt):
    logger = SigoptLogger(name="test_name", project="test_project")
    # without name
    logger.log_image(image=np.ones((255, 255, 3)))
    # with name
    logger.log_image(image=np.ones((255, 255, 3)), name="test_image")
