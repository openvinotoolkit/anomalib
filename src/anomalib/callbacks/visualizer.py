"""Visualizer Callback.

This is assigned by Anomalib Engine internally.
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Any, cast

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib.loggers import AnomalibWandbLogger
from anomalib.loggers.base import ImageLoggerBase
from anomalib.models import AnomalyModule
from anomalib.utils.visualization import (
    BaseVisualizationGenerator,
    GeneratorResult,
    VisualizationStep,
    Visualizer,
)

logger = logging.getLogger(__name__)


class _VisualizationCallback(Callback):
    def __init__(
        self,
        generators: BaseVisualizationGenerator | list[BaseVisualizationGenerator],
        save: bool = False,
        save_root: Path | None = None,
        log: bool = False,
        show: bool = False,
    ) -> None:
        """Callback for visualization that is used internally by the Engine.

        Args:
            generators (BaseVisualizationGenerator | list[BaseVisualizationGenerator]):
                Generator objects that are used for computing the visualizations. Defaults to None.
            save (bool, optional): Save the image. Defaults to False.
            save_root (Path | None, optional): The path to save the images. Defaults to None.
            log (bool, optional): Log the images into the loggers. Defaults to False.
            show (bool, optional): Show the images. Defaults to False.

        Example:
            >>> generators = [ImageVisualizationGenerator(), MetricsVisualizationGenerator()]
            >>> visualization_callback = _VisualizationCallback(
            ... generators=generators,
            ... save=True,
            ... save_root="results/images"
            ... )

            CLI
            $ anomalib fit --model Padim --data MVTec \
                --visualization.generators ImageVisualizationGenerator \
                --visualization.generators+=MetricsVisualizationGenerator

        Raises:
            ValueError: Incase `save_root` is None and `save` is True.
        """
        self.visualizer = Visualizer()
        self.save = save
        if save and save_root is None:
            msg = "`save_root` must be provided if save is True"
            raise ValueError(msg)
        self.save_root: Path = save_root if save_root is not None else Path()  # need this check for mypy
        self.log = log
        self.show = show
        self.generators = generators if isinstance(generators, list) else [generators]

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        for generator in self.generators:
            if generator.visualize_on == VisualizationStep.BATCH:
                for result in generator(
                    trainer=trainer,
                    pl_module=pl_module,
                    outputs=outputs,
                    batch=batch,
                    batch_idx=batch_idx,
                    dataloader_idx=dataloader_idx,
                ):
                    if self.save or self.show:
                        assert result.file_name is not None, "File name is None"
                        if self.save:
                            self.visualizer.save(image=result.image, file_path=self.save_root / result.file_name)
                        if self.show:
                            self.visualizer.show(image=result.image, title=str(self.save_root / result.file_name))
                    if self.log:
                        self._add_to_logger(result, pl_module, trainer)

    def on_test_end(self, trainer: Trainer, pl_module: AnomalyModule) -> None:
        for generator in self.generators:
            if generator.visualize_on == VisualizationStep.STAGE_END:
                generator(trainer=trainer, pl_module=pl_module)

        for logger in trainer.loggers:
            if isinstance(logger, AnomalibWandbLogger):
                logger.save()

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self.on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_predict_end(self, trainer: Trainer, pl_module: AnomalyModule) -> None:
        return self.on_test_end(trainer, pl_module)

    def _add_to_logger(
        self,
        result: GeneratorResult,
        module: AnomalyModule,
        trainer: Trainer,
    ) -> None:
        """Add image to logger.

        Args:
            result (GeneratorResult): Output from the generators.
            module (AnomalyModule): LightningModule from which the global step is extracted.
            trainer (Trainer): Trainer object.
        """
        # Store names of logger and the logger in a dict
        available_loggers = {
            type(logger).__name__.lower().replace("logger", "").replace("anomalib", ""): logger
            for logger in trainer.loggers
        }
        # save image to respective logger
        assert result.file_name is not None, "File name is None"
        filename = result.file_name
        image = result.image
        for log_to in available_loggers:
            # check if logger object is same as the requested object
            if isinstance(available_loggers[log_to], ImageLoggerBase):
                logger: ImageLoggerBase = cast(ImageLoggerBase, available_loggers[log_to])  # placate mypy
                _name = filename.parent.name + "_" + filename.name if isinstance(filename, Path) else filename
                logger.add_image(
                    image=image,
                    name=_name,
                    global_step=module.global_step,
                )
