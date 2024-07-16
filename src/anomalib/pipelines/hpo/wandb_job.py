"""HPO job."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from tempfile import TemporaryDirectory

from anomalib.data import get_datamodule
from anomalib.engine import Engine
from anomalib.models import get_model
from anomalib.pipelines.components import Job
from anomalib.utils.exceptions import try_import
from anomalib.utils.logging import hide_output

from .utils import flatten_hpo_params, set_in_nested_dict

if try_import("wandb"):
    import wandb
    from lightning.pytorch.loggers import WandbLogger

    WAND_AVAILABLE = True
else:
    WAND_AVAILABLE = False


class WandbHPOJob(Job):
    """Wandb based HPO job.

    Args:
        project (str): Name of the project on wandb.
        entity (str): Name of the entity on wandb. This is the team/account name.
        sweep_configuration (dict): Configuration for the sweep. See wandb docs for more info on this configuration.
        count (int): Number of runs to execute.
    """

    name = "wandb_hpo"

    def __init__(self, project: str, entity: str, sweep_configuration: dict, count: int) -> None:
        if not WAND_AVAILABLE:
            msg = "HPO using wandb is requested but wandb is not installed."
            raise ImportError(msg)
        self.original_config = sweep_configuration.copy()
        flattened_hpo_parameters = flatten_hpo_params(sweep_configuration["parameters"])
        sweep_configuration["parameters"] = flattened_hpo_parameters
        self.sweep_configuration = sweep_configuration
        self.sweep_id = wandb.sweep(sweep_configuration, project=project, entity=entity)
        self.count = count

    @hide_output
    def run(self, task_id: int | None = None) -> None:
        """Run the HPO."""
        del task_id  # Not needed for this job

        wandb.agent(self.sweep_id, function=self.train, count=self.count)

    def train(self) -> None:
        """Training method that is called by ``wandb.agent``."""
        wandb_logger = WandbLogger(config=self.sweep_configuration)
        experiment_config = wandb_logger.experiment.config

        run_config = set_in_nested_dict(self.original_config["parameters"], experiment_config.as_dict())

        model = get_model(run_config.get("model"))
        datamodule = get_datamodule(run_config.get("data"))
        with TemporaryDirectory() as temp_dir:
            engine = Engine(
                default_root_dir=temp_dir,
                logger=wandb_logger,
            )
            engine.fit(model=model, datamodule=datamodule)

    @staticmethod
    def collect(_: list) -> None:
        """Does not collect any results."""
        return

    @staticmethod
    def save(_: None = None) -> None:
        """Does not save any results."""
        return
