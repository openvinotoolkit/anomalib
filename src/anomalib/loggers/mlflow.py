"""Tensorboard logger with add image interface."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import os
import numpy as np
from matplotlib.figure import Figure

try:
    from lightning.pytorch.loggers.mlflow import MLFlowLogger
except ModuleNotFoundError:
    print("To use MLFlow logger install it using `pip install mlflow`")
from lightning.pytorch.utilities import rank_zero_only

from .base import ImageLoggerBase


class AnomalibMLFlowLogger(ImageLoggerBase, MLFlowLogger):
    """Logger for tensorboard.

    Adds interface for `add_image` in the logger rather than calling the experiment object.

    .. note::
        Same as the MLFlow Logger provided by PyTorch Lightning and the doc string is reproduced below.

    Logs are saved to
    ``os.path.join(save_dir, name, version)``. This is the default logger in Lightning, it comes
    preinstalled.

    Example:
        >>> from anomalib.engine import Engine
        >>> from anomalib.loggers import AnomalibMLFlowLogger
        ...
        >>> logger = AnomalibMLFlowLogger("fl_logs", name="my_model")
        >>> engine =  Engine(logger=logger)

    Args:
        experiment_name: The name of the experiment.
        run_name: Name of the new run. The `run_name` is internally stored as a ``mlflow.runName`` tag.
            If the ``mlflow.runName`` tag has already been set in `tags`, the value is overridden by the `run_name`.
        tracking_uri: Address of local or remote tracking server.
            If not provided, defaults to `MLFLOW_TRACKING_URI` environment variable if set, otherwise it falls
            back to `file:<save_dir>`.
        tags: A dictionary tags for the experiment.
        save_dir: A path to a local directory where the MLflow runs get saved.
            Defaults to `./mlruns` if `tracking_uri` is not provided.
            Has no effect if `tracking_uri` is provided.
        log_model: Log checkpoints created by :class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint`
            as MLFlow artifacts.

            * if ``log_model == 'all'``, checkpoints are logged during training.
            * if ``log_model == True``, checkpoints are logged at the end of training, except when
              :paramref:`~lightning.pytorch.callbacks.Checkpoint.save_top_k` ``== -1``
              which also logs every checkpoint during training.
            * if ``log_model == False`` (default), no checkpoint is logged.

        prefix: A string to put at the beginning of metric keys.
        artifact_location: The location to store run artifacts. If not provided, the server picks an appropriate
            default.
        run_id: The run identifier of the experiment. If not provided, a new run is started.

    Raises:
        ModuleNotFoundError:
            If required MLFlow package is not installed on the device.

    """

    def __init__(
        self,
        experiment_name: str = "anomalib_logs",
        run_name: str | None = None,
        save_dir: str | None = "./mlruns",
        tracking_uri: str | None = os.getenv("MLFLOW_TRACKING_URI"),
        tags: str | None = None,
        log_model: str | bool = False,
        artifact_location: str | None = None,
        prefix: str = "",
        run_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            experiment_name=experiment_name,
            run_name=run_name,
            save_dir=save_dir,
            tracking_uri=tracking_uri,
            tags=tags,
            log_model=log_model,
            artifact_location=artifact_location,
            prefix=prefix,
            run_id=run_id,
            **kwargs,
        )
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    @rank_zero_only
    def add_image(self, image: np.ndarray | Figure, name: str | None = None, **kwargs) -> None:
        """Interface to add image to MLFlow logger.

        Args:
            image (np.ndarray | Figure): Image to log
            name (str | None): The tag of the image
                Defaults to ``None``.
            kwargs: Accepts only `global_step` (int). The step at which to log the image.
        """
        del kwargs

        if isinstance(image, Figure):
            self.experiment.log_figure(run_id=self.run_id, figure=image, artifact_file=name)  # , **kwargs)
        else:
            self.experiment.log_image(run_id=self.run_id, image=image, artifact_file=name)  # , **kwargs)
