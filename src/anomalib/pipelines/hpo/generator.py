"""HPO sweep job generator."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.pipelines.components import JobGenerator
from anomalib.pipelines.types import PREV_STAGE_RESULT
from anomalib.utils.exceptions import try_import
from anomalib.utils.logging import hide_output

from .job import HPOJob
from .utils import flatten_hpo_params, set_in_nested_dict

if TYPE_CHECKING:
    from lightning.pytorch.loggers import Logger

logger = logging.getLogger(__name__)


class HPOBackend(Enum):
    """HPO backend."""

    COMET = "comet"
    WANDB = "wandb"

    def __str__(self) -> str:
        """Return the string representation."""
        return self.value


@dataclass
class ParserConfig:
    """Common return type for all backend parsers."""

    model_config: dict | None
    data_config: str | None
    logger: "Logger"


class CometParser:
    """Parse the config for Comet.

    Args:
        project (str): The project name.
        entity (str | None): The entity or team name that is associated with the backend account.
    """

    def __init__(self, project: str, entity: str | None = None) -> None:
        self.project = project
        self.entity = entity
        if not try_import("comet_ml"):
            msg = "HPO using comet_ml is requested but comet_ml is not installed."
            raise ImportError(msg)

    def __call__(self, config: dict) -> Generator:
        """Yield experiments."""
        import comet_ml
        from lightning.pytorch.loggers import CometLogger

        # flatten all nested non-hpo keys under parameters
        flattened_dict = flatten_hpo_params(config["parameters"])
        optimizer_config = {
            "algorithm": config["algorithm"],
            "spec": config["spec"],
            "parameters": flattened_dict,
            "name": self.project,
        }
        optimizer = comet_ml.Optimizer(config=optimizer_config)
        for experiment in optimizer.get_experiments(project_name=self.project):
            experiment_logger = CometLogger(workspace=self.entity)
            # allow Lightning logger to use the experiment from optimizer
            experiment_logger._experiment = experiment  # noqa: SLF001
            run_params = set_in_nested_dict(config["parameters"], experiment.params)
            yield ParserConfig(
                model_config=run_params.get("model"),
                data_config=run_params.get("data"),
                logger=experiment_logger,
            )


class WandbParser:
    """Parse the config for Wandb."""


class HPOJobGenerator(JobGenerator):
    """Generate HPOJob.

    Args:
        backend (str | HPOBackend): The backend to use.
        project (str): The project name.
        entity (str | None): The entity or team name that is associated with the backend account.
    """

    def __init__(self, backend: str | HPOBackend, project: str, entity: str | None = None) -> None:
        self.project = project
        self.entity = entity
        self.config_parser = self._get_config_parser(backend)

    def _get_config_parser(self, backend: str | HPOBackend) -> CometParser | WandbParser:
        backend = HPOBackend(backend)
        match backend:
            case HPOBackend.COMET:
                return CometParser(project=self.project, entity=self.entity)
            case HPOBackend.WANDB:
                msg = "wandb backend is not supported yet."
                raise ValueError(msg)

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return HPOJob

    @hide_output
    def generate_jobs(
        self,
        config: dict | None,
        previous_stage_result: PREV_STAGE_RESULT = None,
    ) -> Generator[HPOJob, None, None]:
        """Return iterator based on the arguments."""
        del previous_stage_result  # Not needed for this job
        assert config is not None, "Config is required for HPOJobGenerator."
        if isinstance(self.config_parser, CometParser):
            # temporary as only comet is supported
            for _container in self.config_parser(config):
                yield HPOJob(
                    model=get_model(_container.model_config),
                    datamodule=get_datamodule(_container.data_config),
                    logger=_container.logger,
                )
