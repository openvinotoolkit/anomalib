"""Compare job."""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product

import pandas as pd
from jsonargparse import ArgumentParser, Namespace

from anomalib.pipelines.jobs.base import Job
from anomalib.pipelines.utils import hide_output


@dataclass
class ModelResult:
    """Model result."""

    name: str
    result: float


class CompareJob(Job):
    """Compare models."""

    name = "compare"

    @hide_output
    def run(self, model1: ModelResult, model2: ModelResult, task_id: int) -> dict[str, float]:
        """Read model outputs from disk and compare them."""
        del task_id  # not used
        score = self._dummy_comparison(model1.result, model2.result)
        output = {"test": f"{model1.name} vs {model2.name}", "score": score}
        self.logger.info(f"Comparison between {model1.name} and {model2.name} done.")
        return output

    def collect(self, results: list[dict[str, float]]) -> pd.DataFrame:
        """Gather the results returned from run."""
        output: dict[str, float] = {}
        for key in results[0]:
            output[key] = []
        for result in results:
            for key, value in result.items():
                output[key].append(value)
        return pd.DataFrame(output)

    def save(self, results: pd.DataFrame) -> None:
        results.to_csv("comparison_results.csv", index=False)

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        """This stage does not require arguments."""

    @staticmethod
    def get_iterator(args: Namespace | None = None) -> Iterator:
        """Iterate over combinations."""
        del args  # args are not needed
        results = pd.read_csv("fit.csv")
        model_result_list = [
            ModelResult(name=f'{row["model"]}_{row["backbone"]}', result=row["result"]) for _, row in results.iterrows()
        ]
        combinations = list(product(model_result_list, model_result_list))
        for combination in combinations:
            yield {"model1": combination[0], "model2": combination[1]}

    def _dummy_comparison(self, result1: float, result2: float) -> float:
        """Dummy comparison function."""
        return result1 - result2
