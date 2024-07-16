"""HPO sweep job generator."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Generator

from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.pipelines.components import JobGenerator
from anomalib.pipelines.components.base.job import Job
from anomalib.pipelines.types import PREV_STAGE_RESULT
from anomalib.utils.exceptions import try_import
from anomalib.utils.logging import hide_output

from .job import CometHPOJob, WandbHPOJob
from .utils import flatten_hpo_params, set_in_nested_dict

logger = logging.getLogger(__name__)


class WandbHPOJobGenerator(JobGenerator):
    """Generate Wandb based HPO job."""

    def __init__(self) -> None:
        if not try_import("wandb"):
            msg = "HPO using wandb is requested but wandb is not installed."
            raise ImportError(msg)

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return WandbHPOJob

    @hide_output
    def generate_jobs(
        self,
        config: dict,
        prev_stage_result: PREV_STAGE_RESULT,
    ) -> Generator[Job, None, None]:
        """Generate a single job that uses ``wandb.agent`` internally."""
        del prev_stage_result  # Not needed for this job
        yield WandbHPOJob(
            project=config["project"],
            entity=config["entity"],
            sweep_configuration=config["sweep_configuration"],
            count=config["count"],
        )


class CometHPOJobGenerator(JobGenerator):
    """Generate Comet based HPO job."""

    def __init__(self) -> None:
        if not try_import("comet_ml"):
            msg = "HPO using comet_ml is requested but comet_ml is not installed."
            raise ImportError(msg)

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return CometHPOJob

    @hide_output
    def generate_jobs(
        self,
        config: dict,
        prev_stage_result: PREV_STAGE_RESULT,
    ) -> Generator[Job, None, None]:
        """Generate jobs based on parameters suggested from comet hyperparameter optimizer."""
        import comet_ml
        from lightning.pytorch.loggers import CometLogger

        del prev_stage_result  # Not needed for this job
        # flatten all nested non-hpo keys under parameters
        flattened_dict = flatten_hpo_params(config["parameters"])
        optimizer_config = {
            "algorithm": config["algorithm"],
            "spec": config["spec"],
            "parameters": flattened_dict,
            "name": config["name"],
        }
        optimizer = comet_ml.Optimizer(config=optimizer_config)
        for experiment in optimizer.get_experiments(project_name=config["name"]):
            experiment_logger = CometLogger(workspace=config["workspace"])
            # allow Lightning logger to use the experiment from optimizer
            experiment_logger._experiment = experiment  # noqa: SLF001
            run_params = set_in_nested_dict(config["parameters"], experiment.params)
            yield CometHPOJob(
                model=get_model(run_params.get("model")),
                datamodule=get_datamodule(run_params.get("data")),
                logger=experiment_logger,
            )
