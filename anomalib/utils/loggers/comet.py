"""comet logger with add image interface."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

import numpy as np
from matplotlib.figure import Figure
from pytorch_lightning.loggers.comet import CometLogger
from pytorch_lightning.utilities import rank_zero_only


from .base import ImageLoggerBase


class AnomalibCometLogger(ImageLoggerBase, CometLogger):

    def __init__(
        self,
        api_key: Optional[str] = None,
        save_dir: Optional[str] = None,
        project_name: Optional[str] = None,
        rest_api_key: Optional[str] = None,
        experiment_name: Optional[str] = None,
        experiment_key: Optional[str] = None,
        offline: bool = False,
        prefix: str = "",
        **kwargs
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
            **kwargs
        )
       

    @rank_zero_only
    def add_image(self, image: Union[np.ndarray, Figure], name: Optional[str] = None, **kwargs: Any):
        """Interface to add image to comet logger.

        Args:
            image (Union[np.ndarray, Figure]): Image to log
            name (Optional[str]): The tag of the image
            kwargs: Accepts only `global_step` (int). The step at which to log the image.
        """
        if "global_step" not in kwargs:
            raise ValueError("`global_step` is required for tensorboard logger")

        global_step = kwargs["global_step"]
        # Need to call different functions of `Experiment` for  Figure vs np.ndarray

        if isinstance(image, Figure):
            self.experiment.log_figure(figure_name=name, figure=image, step=global_step)
        else:
            self.experiment.log_image(name=name, image_data=image, step=global_step)