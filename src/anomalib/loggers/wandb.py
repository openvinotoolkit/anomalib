"""wandb logger with add image interface."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Literal

import numpy as np
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from matplotlib.figure import Figure

from anomalib.utils.exceptions import try_import

from .base import ImageLoggerBase

if try_import("wandb"):
    import wandb

if TYPE_CHECKING:
    from wandb.sdk.lib import RunDisabled
    from wandb.sdk.wandb_run import Run


class AnomalibWandbLogger(ImageLoggerBase, WandbLogger):
    """Logger for wandb.

    Adds interface for `add_image` in the logger rather than calling the experiment object.

    .. note::
        Same as the wandb Logger provided by PyTorch Lightning and the doc string is reproduced below.

    Log using `Weights and Biases <https://www.wandb.com/>`_.

    Install it with pip:

    .. code-block:: bash

        $ pip install wandb

    Args:
        name: Display name for the run.
            Defaults to ``None``.
        save_dir: Path where data is saved (wandb dir by default).
            Defaults to ``None``.
        version: Sets the version, mainly used to resume a previous run.
        offline: Run offline (data can be streamed later to wandb servers).
            Defaults to ``False``.
        dir: Alias for save_dir.
        id: Sets the version, mainly used to resume a previous run.
            Defaults to ``None``.
        anonymous: Enables or explicitly disables anonymous logging.
            Defaults to ``None``.
        version: Same as id.
            Defaults to ``None``.
        project: The name of the project to which this run will belong.
            Defaults to ``None``.
        log_model: Save checkpoints in wandb dir to upload on W&B servers.
            Defaults to ``False``.
        experiment: WandB experiment object. Automatically set when creating a run.
            Defaults to ``None``.
        prefix: A string to put at the beginning of metric keys.
            Defaults to ``''``.
        **kwargs: Arguments passed to :func:`wandb.init` like `entity`, `group`, `tags`, etc.

    Raises:
        ImportError:
            If required WandB package is not installed on the device.
        MisconfigurationException:
            If both ``log_model`` and ``offline``is set to ``True``.

    Example:
        >>> from anomalib.loggers import AnomalibWandbLogger
        >>> from anomalib.engine import Engine
        ...
        >>> wandb_logger = AnomalibWandbLogger()
        >>> engine =  Engine(logger=wandb_logger)

    .. note::
        When logging manually through `wandb.log` or `trainer.logger.experiment.log`,
        make sure to use `commit=False` so the logging step does not increase.

    See Also:
        - `Tutorial <https://colab.research.google.com/drive/16d1uctGaw2y9KhGBlINNTsWpmlXdJwRW?usp=sharing>`__
          on how to use W&B with PyTorch Lightning
        - `W&B Documentation <https://docs.wandb.ai/integrations/lightning>`__

    """

    def __init__(
        self,
        name: str | None = None,
        save_dir: _PATH = ".",
        version: str | None = None,
        offline: bool = False,
        dir: _PATH | None = None,  # kept to match wandb init # noqa: A002
        id: str | None = None,  # kept to match wandb init # noqa: A002
        anonymous: bool | None = None,
        project: str | None = None,
        log_model: Literal["all"] | bool = False,
        experiment: "Run | RunDisabled | None" = None,
        prefix: str = "",
        checkpoint_name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=offline,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )
        self.image_list: list[wandb.Image] = []  # Cache images

    @rank_zero_only
    def add_image(self, image: np.ndarray | Figure, name: str | None = None, **kwargs) -> None:
        """Interface to add image to wandb logger.

        Args:
            image (np.ndarray | Figure): Image to log
            name (str | None): The tag of the image
                Defaults to ``None``.
            kwargs: Additional arguments to `wandb.Image`
        """
        del kwargs  # Unused argument.

        image = wandb.Image(image, caption=name)
        self.image_list.append(image)

    @rank_zero_only
    def save(self) -> None:
        """Upload images to wandb server.

        .. note::
            There is a limit on the number of images that can be logged together to the `wandb` server.
        """
        super().save()
        if len(self.image_list) > 1:
            wandb.log({"Predictions": self.image_list})
            self.image_list = []
            self.image_list = []
