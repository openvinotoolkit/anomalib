"""Extract job."""

import pickle
from collections.abc import Iterator
from pathlib import Path

import torch
from jsonargparse import ArgumentParser, Namespace

from anomalib.pipelines.jobs.base import Job
from anomalib.pipelines.utils import dict_from_namespace, hide_output
from anomalib.pipelines.utils.actions import GridSearchAction, get_iterator_from_grid_dict


class ExtractJob(Job):
    """Extract random feature vectors."""

    name = "extract"

    @hide_output
    def run(self, backbone: str, image: torch.Tensor, task_id: int) -> dict[str, str | torch.Tensor]:
        """Extract feature vectors."""
        del task_id  # not used
        output = {
            "backbone": backbone,
            "feature": torch.rand_like(image).flatten(),
        }
        self.logger.info(f"Feature extracted from {backbone}")
        return output

    def collect(self, results: list[dict[str, str | torch.Tensor]]) -> dict[str, str | torch.Tensor]:
        """Gather the results returned from run."""
        output = {}
        for key in results[0]:
            output[key] = []
        for result in results:
            for key, value in result.items():
                output[key].append(value)
        return output

    def save(self, result: dict[str, str | torch.Tensor]) -> None:
        """Save the results."""
        with Path("feature_vectors.pkl").open("wb") as file:
            pickle.dump(result, file)

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        """Add arguments to the parser."""
        with GridSearchAction.allow_default_instance_context():
            action = parser.add_argument(
                f"--{ExtractJob.name}.backbone",
                help="Backbone to use for feature extraction.",
                action=GridSearchAction(str | list[str]),
            )
        action.sub_add_kwargs = {"fail_untyped": True, "sub_configs": True, "instantiate": True}

    @staticmethod
    def get_iterator(args: Namespace) -> Iterator:
        """Get the iterator for the grid search."""
        container = {
            "backbone": dict_from_namespace(args),
            "image": torch.rand(3, 224, 224),
        }
        return get_iterator_from_grid_dict(container)
