"""Test Models on all MVTec AD Categories."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import multiprocessing
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import seed_everything

from anomalib.utils.sweep.config import flatten_sweep_params
from tests.helpers.dataset import get_dataset_path
from tests.helpers.model import model_load_test, setup_model_train


def get_model_nncf_cat() -> List:
    """Test helper for getting cartesian product of models and categories.

    Returns:
        List: Returns a combination of models with their nncf support for each category.
    """
    model_support = [
        ("cflow", False),
        ("csflow", False),
        ("dfkde", False),
        ("dfm", False),
        ("ganomaly", False),
        # ("stfpm", True),
        ("padim", False),
        ("patchcore", False),
        ("stfpm", False),
    ]
    categories = random.sample(
        [
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
        ],
        k=3,
    )

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
        if "optimization" in config.keys() and "nncf" in config.optimization.keys() and config.optimization.nncf.apply:
            threshold = threshold.nncf
        if not (
            np.isclose(results["image_AUROC"], threshold["image_AUROC"], rtol=0.05)
            or (results["image_AUROC"] >= threshold["image_AUROC"])
        ):
            raise AssertionError(
                f"results['image_AUROC']: {results['image_AUROC']} >= "
                f"threshold['image_AUROC']: {threshold['image_AUROC']}"
            )

        if config.dataset.task == "segmentation":
            if not (
                np.isclose(results["pixel_AUROC"], threshold["pixel_AUROC"], rtol=0.05)
                or (results["pixel_AUROC"] >= threshold["pixel_AUROC"])
            ):
                raise AssertionError(
                    f"results['pixel_AUROC']:{results['pixel_AUROC']} >= "
                    f"threshold['pixel_AUROC']:{threshold['pixel_AUROC']}"
                )
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

    def runner(self, run_configs, path, score_type, device_id):
        for model_name, nncf, category in run_configs:
            try:
                with tempfile.TemporaryDirectory() as project_path:
                    # Fix seed
                    seed_everything(42, workers=True)
                    config, datamodule, model, trainer = setup_model_train(
                        model_name=model_name,
                        dataset_path=path,
                        nncf=nncf,
                        project_path=project_path,
                        category=category,
                        score_type=score_type,
                        device=[device_id],
                    )

                    # test model metrics
                    results = self._test_metrics(trainer=trainer, config=config, model=model, datamodule=datamodule)

                    # test model load
                    model_load_test(config=config, datamodule=datamodule, results=results)

                    self._save_to_csv(config, results)
            except AssertionError as assertion_error:
                raise Exception(f"Model: {model_name} NNCF:{nncf} Category:{category}") from assertion_error

    def test_model(self, path=get_dataset_path(), score_type=None):
        run_configs = get_model_nncf_cat()
        with ProcessPoolExecutor(
            max_workers=torch.cuda.device_count(), mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            jobs = []
            for device_id, run_split in enumerate(
                range(0, len(run_configs), math.ceil(len(run_configs) / torch.cuda.device_count()))
            ):
                jobs.append(
                    executor.submit(
                        self.runner,
                        run_configs[run_split : run_split + math.ceil(len(run_configs) / torch.cuda.device_count())],
                        path,
                        score_type,
                        device_id,
                    )
                )
            for job in jobs:
                try:
                    job.result()
                except Exception as e:
                    raise e
