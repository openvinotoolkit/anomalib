"""MLFlow logger with add image interface."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Literal

import numpy as np
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from lightning.pytorch.utilities import rank_zero_only
from matplotlib.figure import Figure

from .base import ImageLoggerBase


class AnomalibMLFlowLogger(ImageLoggerBase, MLFlowLogger):
    """Logger for MLFlow.

    Adds interface for ``add_image`` in the logger rather than calling the
    experiment object.

    .. note::
        Same as the MLFlowLogger provided by PyTorch Lightning and the doc string is reproduced below.

    Track your parameters, metrics, source code and more using
    `MLFlow <https://mlflow.org/#core-concepts>`_.

    Install it with pip:

    .. code-block:: bash

        pip install mlflow

    Args:
        experiment_name: The name of the experiment.
        run_name: Name of the new run.
            The `run_name` is internally stored as a ``mlflow.runName`` tag.
            If the ``mlflow.runName`` tag has already been set in `tags`, the value is overridden by the `run_name`.
        tracking_uri: Address of local or remote tracking server.
            If not provided, defaults to `MLFLOW_TRACKING_URI` environment variable if set, otherwise it falls
            back to `file:<save_dir>`.
        save_dir: A path to a local directory where the MLflow runs get saved.
            Defaults to `./mlruns` if `tracking_uri` is not provided.
            Has no effect if `tracking_uri` is provided.
        log_model: Log checkpoints created by `ModelCheckpoint` as MLFlow artifacts.

            - if ``log_model == 'all'``, checkpoints are logged during training.
            - if ``log_model == True``, checkpoints are logged at the end of training, \
                except when `save_top_k == -1` which also logs every checkpoint during training.
            - if ``log_model == False`` (default), no checkpoint is logged.

        prefix: A string to put at the beginning of metric keys. Defaults to ``''``.
        kwargs: Additional arguments like `tags`, `artifact_location` etc. used by
            `MLFlowExperiment` can be passed as keyword arguments in this logger.

    Example:
        >>> from anomalib.loggers import AnomalibMLFlowLogger
        >>> from anomalib.engine import Engine
        ...
        >>> mlflow_logger = AnomalibMLFlowLogger()
        >>> engine = Engine(logger=mlflow_logger)

    See Also:
        - `MLFlow Documentation <https://mlflow.org/docs/latest/>`_.
    """

    def __init__(
        self,
        experiment_name: str | None = "anomalib_logs",
        run_name: str | None = None,
        tracking_uri: str | None = os.getenv("MLFLOW_TRACKING_URI"),
        save_dir: str | None = "./mlruns",
        log_model: Literal[True, False, "all"] | None = False,
        prefix: str | None = "",
        **kwargs,
    ) -> None:
        super().__init__(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            save_dir=save_dir,
            log_model=log_model,
            prefix=prefix,
            **kwargs,
        )

    @rank_zero_only
    def add_image(self, image: np.ndarray | Figure, name: str | None = None, **kwargs) -> None:
        """Interface to log images in the mlflow loggers.

        Args:
            image (np.ndarray | Figure): Image to log.
            name (str | None): The tag of the image defaults to ``None``.
            kwargs: Additional keyword arguments that are only used if `image` is of type Figure.
                These arguments are passed directly to the method that saves the figure.
                If `image` is a NumPy array, `kwargs` has no effect.
        """
        # Need to call different functions of `Experiment` for  Figure vs np.ndarray
        if isinstance(image, Figure):
            self.experiment.log_figure(run_id=self.run_id, figure=image, artifact_file=name, **kwargs)
        else:
            self.experiment.log_image(run_id=self.run_id, image=image, artifact_file=name)
