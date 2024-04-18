"""Fit job."""

from collections.abc import Iterator
from pathlib import Path

import joblib
import pandas as pd
import torch
from jsonargparse import ArgumentParser, Namespace

from anomalib.pipelines.jobs.base import Job
from anomalib.pipelines.utils import dict_from_namespace, hide_output
from anomalib.pipelines.utils.actions import GridSearchAction, get_iterator_from_grid_dict


class ModelA(torch.nn.Module):
    """Model A."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.tensor(0.0)


class ModelB(torch.nn.Module):
    """Model B."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.tensor(1.0)


class FitJob(Job):
    """Fit models on the feature vectors."""

    name = "Fit"

    @hide_output
    def run(self, model: torch.nn.Module, feature: torch.Tensor, backbone: str) -> dict[str, float | str]:
        """Fit the model on the feature vectors."""
        output = model(feature)
        output = {"output": output.item(), "model": model.__class__.__name__, "backbone": backbone}
        self.logger.info(f"Model {model.__class__.__name__} fitted on {backbone}")
        return output

    def collect(self, results: list[dict[str, float | str]]) -> pd.DataFrame:
        """Gather the results returned from run."""
        output: dict = {
            "result": [],
            "model": [],
            "backbone": [],
        }
        for result in results:
            output["result"].append(result["output"])
            output["model"].append(result["model"])
            output["backbone"].append(result["backbone"])
        return pd.DataFrame(output)

    def save(self, results: pd.DataFrame) -> None:
        """Save results to disk."""
        results.to_csv(Path("fit.csv"), index=False)

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        """Add arguments to the parser."""
        with GridSearchAction.allow_default_instance_context():
            action = parser.add_argument(
                f"--{FitJob.name}.model",
                help="Model to fit on.",
                action=GridSearchAction(str | list[str]),
            )
        action.sub_add_kwargs = {"fail_untyped": True, "sub_configs": True, "instantiate": True}

    @staticmethod
    def get_iterator(args: Namespace) -> Iterator:
        """Get iterator for the grid search."""
        features = joblib.load("feature_vectors.pkl")

        container = {
            "model": dict_from_namespace(args.model),
            "feature": {"grid": features["feature"]},
            "backbone": {
                "grid": [value for backbone_dict in features["backbone"] for _, value in backbone_dict.items()],
            },
        }
        iterator = get_iterator_from_grid_dict(container)
        for config in iterator:
            yield {
                "model": ModelA() if config["model"] == "A" else ModelB(),
                "feature": config["feature"],
                "backbone": config["backbone"],
            }
