"""Comet logger with image logging capabilities.

This module provides a Comet logger implementation that adds an interface for
logging images. It extends both the base image logger and PyTorch Lightning's
Comet logger.

Example:
    >>> from anomalib.loggers import AnomalibCometLogger
    >>> from anomalib.engine import Engine
    >>> comet_logger = AnomalibCometLogger()  # doctest: +SKIP
    >>> engine = Engine(logger=comet_logger)  # doctest: +SKIP

    Log an image:
    >>> import numpy as np
    >>> image = np.random.rand(32, 32, 3)  # doctest: +SKIP
    >>> comet_logger.add_image(
    ...     image=image,
    ...     name="test_image",
    ...     global_step=0
    ... )  # doctest: +SKIP
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from matplotlib.figure import Figure

try:
    from lightning.pytorch.loggers.comet import CometLogger
except ModuleNotFoundError:
    print("To use comet logger install it using `pip install comet-ml`")
from lightning.pytorch.utilities import rank_zero_only

from .base import ImageLoggerBase


class AnomalibCometLogger(ImageLoggerBase, CometLogger):
    """Logger for Comet ML with image logging capabilities.

    This logger extends PyTorch Lightning's CometLogger with an interface for
    logging images. It inherits from both :class:`ImageLoggerBase` and
    :class:`CometLogger`.

    Args:
        api_key: API key found on Comet.ml. If not provided, will be loaded from
            ``COMET_API_KEY`` environment variable or ``~/.comet.config``.
            Required for online mode.
            Defaults to ``None``.
        save_dir: Directory path to save local comet logs. Required for offline
            mode. Also sets checkpoint directory if provided.
            Defaults to ``None``.
        project_name: Project name for the experiment. Creates new project if
            doesn't exist.
            Defaults to ``None``.
        rest_api_key: Rest API key from Comet.ml settings. Used for version
            tracking.
            Defaults to ``None``.
        experiment_name: Name for this experiment on Comet.ml.
            Defaults to ``None``.
        experiment_key: Key to restore existing experiment.
            Defaults to ``None``.
        offline: Force offline mode even with API key. Useful when using
            ``save_dir`` for checkpoints with ``~/.comet.config``.
            Defaults to ``False``.
        prefix: String to prepend to metric keys.
            Defaults to ``""``.
        **kwargs: Additional arguments passed to :class:`CometExperiment`
            (e.g. ``workspace``, ``log_code``).

    Raises:
        ModuleNotFoundError: If ``comet-ml`` package is not installed.
        MisconfigurationException: If neither ``api_key`` nor ``save_dir``
            provided.

    Example:
        >>> from anomalib.loggers import AnomalibCometLogger
        >>> comet_logger = AnomalibCometLogger(
        ...     project_name="anomaly_detection"
        ... )  # doctest: +SKIP

    Note:
        For more details, see the `Comet Documentation
        <https://www.comet.com/docs/v2/integrations/ml-frameworks/pytorch-lightning/>`_
    """

    def __init__(
        self,
        api_key: str | None = None,
        save_dir: str | None = None,
        project_name: str | None = None,
        rest_api_key: str | None = None,
        experiment_name: str | None = None,
        experiment_key: str | None = None,
        offline: bool = False,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__(
            api_key=api_key,
            save_dir=save_dir,
            project_name=project_name,
            rest_api_key=rest_api_key,
            experiment_name=experiment_name,
            experiment_key=experiment_key,
            offline=offline,
            prefix=prefix,
            **kwargs,
        )
        self.experiment.log_other("Created from", "Anomalib")

    @rank_zero_only
    def add_image(self, image: np.ndarray | Figure, name: str | None = None, **kwargs) -> None:
        """Log an image to Comet.

        Args:
            image: Image to log, either numpy array or matplotlib figure.
            name: Name/tag for the image.
                Defaults to ``None``.
            **kwargs: Must contain ``global_step`` (int) indicating the step at
                which to log the image.

        Raises:
            ValueError: If ``global_step`` not provided in kwargs.

        Example:
            >>> import numpy as np
            >>> from matplotlib.figure import Figure
            >>> logger = AnomalibCometLogger()  # doctest: +SKIP
            >>> # Log numpy array
            >>> image_array = np.random.rand(32, 32, 3)  # doctest: +SKIP
            >>> logger.add_image(
            ...     image=image_array,
            ...     name="test_image",
            ...     global_step=0
            ... )  # doctest: +SKIP
            >>> # Log matplotlib figure
            >>> fig = Figure()  # doctest: +SKIP
            >>> logger.add_image(
            ...     image=fig,
            ...     name="test_figure",
            ...     global_step=1
            ... )  # doctest: +SKIP
        """
        if "global_step" not in kwargs:
            msg = "`global_step` is required for comet logger"
            raise ValueError(msg)

        global_step = kwargs["global_step"]
        # Need to call different functions of `Experiment` for  Figure vs np.ndarray

        if isinstance(image, Figure):
            self.experiment.log_figure(figure_name=name, figure=image, step=global_step)
        else:
            self.experiment.log_image(name=name, image_data=image, step=global_step)
