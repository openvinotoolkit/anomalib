"""comet logger with add image interface."""

# Copyright (C) 2022-2024 Intel Corporation
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
    """Logger for comet.

    Adds interface for ``add_image`` in the logger rather than calling the
    experiment object.

    .. note::
        Same as the CometLogger provided by PyTorch Lightning and the doc string
        is reproduced below.

    Track your parameters, metrics, source code and more using
    `Comet <https://www.comet.com/site/products/ml-experiment-tracking/?utm_source=anomalib&utm_medium=referral>`_.

    Install it with pip:

    .. code-block:: bash

        pip install comet-ml

    Comet requires either an API Key (online mode) or a local directory path
    (offline mode).

    Args:
        api_key: Required in online mode. API key, found on Comet.ml. If not
            given, this will be loaded from the environment variable
            COMET_API_KEY or ~/.comet.config if either exists.
            Defaults to ``None``.
        save_dir: Required in offline mode. The path for the directory to save
            local comet logs. If given, this also sets the directory for saving
            checkpoints.
            Defaults to ``None``.
        project_name: Optional. Send your experiment to a specific project.
            Otherwise will be sent to Uncategorized Experiments.
            If the project name does not already exist, Comet.ml will create a
            new project.
            Defaults to ``None``.
        rest_api_key: Optional. Rest API key found in Comet.ml settings.
            This is used to determine version number
            Defaults to ``None``.
        experiment_name: Optional. String representing the name for this
            particular experiment on Comet.ml.
            Defaults to ``None``.
        experiment_key: Optional. If set, restores from existing experiment.
            Defaults to ``None``.
        offline: If api_key and save_dir are both given, this determines whether
            the experiment will be in online or offline mode. This is useful if
            you use save_dir to control the checkpoints directory and have a
            ~/.comet.config file but still want to run offline experiments.
            Defaults to ``None``.
        prefix: A string to put at the beginning of metric keys.
            Defaults to ``""``.
        kwargs: Additional arguments like `workspace`, `log_code`, etc. used by
            :class:`CometExperiment` can be passed as keyword arguments in this
            logger.

    Raises:
        ModuleNotFoundError:
            If required Comet package is not installed on the device.
        MisconfigurationException:
            If neither ``api_key`` nor ``save_dir`` are passed as arguments.

    Example:
        >>> from anomalib.loggers import AnomalibCometLogger
        >>> from anomalib.engine import Engine
        ...
        >>> comet_logger = AnomalibCometLogger()
        >>> engine =  Engine(logger=comet_logger)

    See Also:
        - `Comet Documentation <https://www.comet.com/docs/v2/integrations/ml-frameworks/pytorch-lightning/>`__
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
        """Interface to add image to comet logger.

        Args:
            image (np.ndarray | Figure): Image to log.
            name (str | None): The tag of the image
                Defaults to ``None``.
            kwargs: Accepts only `global_step` (int). The step at which to log the image.
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
