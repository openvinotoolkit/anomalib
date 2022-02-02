"""Test Models on all MVTec Categories."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import itertools
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import pytest
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import seed_everything

from anomalib.utils.hpo.config import flatten_sweep_params
from tests.helpers.dataset import get_dataset_path
from tests.helpers.model import model_load_test, setup


def get_model_nncf_cat() -> List:
    """Test helper for getting cartesian product of models and categories.

    Returns:
        List: Returns a combination of models with their nncf support for each category.
    """
    model_support = [
        ("padim", False),
        ("dfkde", False),
        ("dfm", False),
        ("stfpm", False),
        ("stfpm", True),
        ("patchcore", False),
        ("cflow", False),
    ]
    categories = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

    return [
        (model, nncf, category) for ((model, nncf), category) in list(itertools.product(*[model_support, categories]))
    ]


class TestModel:
    """Run Model on all categories."""

    def _test_metrics(self, trainer, config, model, datamodule):
        """Tests the model metrics but also acts as a setup."""

        results = trainer.test(model=model, datamodule=datamodule)[0]

        thresholds = OmegaConf.load("tests/nightly/models/performance_thresholds.yaml")

        threshold = thresholds[config.model.name][config.dataset.category]
        if "optimization" in config.keys() and config.optimization.nncf.apply:
            threshold = threshold.nncf
        assert results["image_AUROC"] >= threshold["image_AUROC"]

        if config.dataset.task == "segmentation":
            assert results["pixel_AUROC"] >= threshold["pixel_AUROC"]
        return results

    def _save_to_csv(self, config: Union[DictConfig, ListConfig], results: Dict):
        """Save model results to csv. Useful for tracking model drift.

        Args:
            config (Union[DictConfig, ListConfig]): Model config which is also added to csv for complete picture.
            results (Dict): Metrics from trainer.test
        """
        # Save results in csv for tracking model drift
        model_metrics = flatten_sweep_params(config)
        # convert dict, list values to string
        for key, val in model_metrics.items():
            if isinstance(val, (list, dict, ListConfig, DictConfig)):
                model_metrics[key] = str(val)
        for metric, value in results.items():
            model_metrics[metric] = value
        model_metrics_df = pd.DataFrame([model_metrics])

        result_path = Path(f"tests/artifacts/{datetime.now().strftime('%m_%d_%Y')}.csv")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        if not result_path.is_file():
            model_metrics_df.to_csv(result_path)
        else:
            model_metrics_df.to_csv(result_path, mode="a", header=False)

    @pytest.mark.parametrize(["model_name", "nncf", "category"], get_model_nncf_cat())
    def test_model(self, model_name, nncf, category, path=get_dataset_path(), score_type=None):
        # Fix seed
        seed_everything(42)

        with tempfile.TemporaryDirectory() as project_path:
            config, datamodule, model, trainer = setup(
                model_name=model_name,
                dataset_path=path,
                nncf=nncf,
                project_path=project_path,
                category=category,
                score_type=score_type,
            )

            # test model metrics
            results = self._test_metrics(trainer=trainer, config=config, model=model, datamodule=datamodule)

            # test model load
            model_load_test(config=config, datamodule=datamodule, results=results)

            self._save_to_csv(config, results)
