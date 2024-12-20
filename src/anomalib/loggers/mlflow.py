"""MLFlow logger with image logging capabilities.

This module provides an MLFlow logger implementation that adds an interface for
logging images. It extends both the base image logger and PyTorch Lightning's
MLFlow logger.

Example:
    >>> from anomalib.loggers import AnomalibMLFlowLogger
    >>> from anomalib.engine import Engine
    >>> mlflow_logger = AnomalibMLFlowLogger()
    >>> engine = Engine(logger=mlflow_logger)  # doctest: +SKIP

    Log an image:
    >>> import numpy as np
    >>> image = np.random.rand(32, 32, 3)  # doctest: +SKIP
    >>> mlflow_logger.add_image(
    ...     image=image,
    ...     name="test_image"
    ... )  # doctest: +SKIP
"""

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
    """Logger for MLFlow with image logging capabilities.

    This logger extends PyTorch Lightning's MLFlowLogger with an interface for
    logging images. It inherits from both :class:`ImageLoggerBase` and
    :class:`MLFlowLogger`.

    Args:
        experiment_name: Name of the experiment. If not provided, defaults to
            ``"anomalib_logs"``.
        run_name: Name of the new run. The ``run_name`` is internally stored as
            a ``mlflow.runName`` tag. If the ``mlflow.runName`` tag has already
            been set in ``tags``, the value is overridden by the ``run_name``.
        tracking_uri: Address of local or remote tracking server. If not provided,
            defaults to ``MLFLOW_TRACKING_URI`` environment variable if set,
            otherwise falls back to ``file:<save_dir>``.
        save_dir: Path to local directory where MLflow runs are saved. Defaults
            to ``"./mlruns"`` if ``tracking_uri`` is not provided. Has no effect
            if ``tracking_uri`` is provided.
        log_model: Log checkpoints created by ``ModelCheckpoint`` as MLFlow
            artifacts:

            - if ``"all"``: checkpoints are logged during training
            - if ``True``: checkpoints are logged at end of training (except when
              ``save_top_k == -1`` which logs every checkpoint during training)
            - if ``False`` (default): no checkpoints are logged

        prefix: String to prepend to metric keys. Defaults to ``""``.
        **kwargs: Additional arguments like ``tags``, ``artifact_location`` etc.
            used by ``MLFlowExperiment``.

    Example:
        >>> from anomalib.loggers import AnomalibMLFlowLogger
        >>> from anomalib.engine import Engine
        >>> mlflow_logger = AnomalibMLFlowLogger(
        ...     experiment_name="my_experiment",
        ...     run_name="my_run"
        ... )  # doctest: +SKIP
        >>> engine = Engine(logger=mlflow_logger)  # doctest: +SKIP

    See Also:
        - `MLFlow Documentation <https://mlflow.org/docs/latest/>`_
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
        """Log images to MLflow.

        Args:
            image: Image to log, can be either a numpy array or matplotlib Figure.
            name: Name/title of the image. Defaults to ``None``.
            **kwargs: Additional keyword arguments passed to the MLflow logging
                method when ``image`` is a Figure. Has no effect when ``image``
                is a numpy array.
        """
        # Need to call different functions of `Experiment` for  Figure vs np.ndarray
        if isinstance(image, Figure):
            self.experiment.log_figure(run_id=self.run_id, figure=image, artifact_file=name, **kwargs)
        else:
            self.experiment.log_image(run_id=self.run_id, image=image, artifact_file=name)
