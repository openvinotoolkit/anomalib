"""Utility functions to compare AUPIMO scores with the benchmark results from AUPIMO's official repository.

Official repository: https://github.com/jpcbertoldo/aupimo
Benchmark data: https://github.com/jpcbertoldo/aupimo/tree/main/data/experiments/benchmark
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import pandas as pd
import requests
import torch
from pandas import DataFrame

from .dataclasses import AUPIMOResult

logger = logging.getLogger(__name__)

AUPIMO_BENCHMARK_MODELS = {
    "efficientad_wr101_m_ext",
    "efficientad_wr101_s_ext",
    "fastflow_cait_m48_448",
    "fastflow_wr50",
    "padim_r18",
    "padim_wr50",
    "patchcore_wr101",
    "patchcore_wr50",
    "pyramidflow_fnf_ext",
    "pyramidflow_r18_ext",
    "rd++_wr50_ext",
    "simplenet_wr50_ext",
    "uflow_ext",
}

AUPIMO_BENCHMARK_DATASETS = {
    "mvtec/bottle",
    "mvtec/cable",
    "mvtec/capsule",
    "mvtec/carpet",
    "mvtec/grid",
    "mvtec/hazelnut",
    "mvtec/leather",
    "mvtec/metal_nut",
    "mvtec/pill",
    "mvtec/screw",
    "mvtec/tile",
    "mvtec/toothbrush",
    "mvtec/transistor",
    "mvtec/wood",
    "mvtec/zipper",
    "visa/candle",
    "visa/capsules",
    "visa/cashew",
    "visa/chewinggum",
    "visa/fryum",
    "visa/macaroni1",
    "visa/macaroni2",
    "visa/pcb1",
    "visa/pcb2",
    "visa/pcb3",
    "visa/pcb4",
    "visa/pipe_fryum",
}


def _validate_benchmark_model(model: str) -> None:
    if model not in AUPIMO_BENCHMARK_MODELS:
        msg = f"Model '{model}' not available. Choose one of {sorted(AUPIMO_BENCHMARK_MODELS)}."
        raise ValueError(msg)


def _validate_benchmark_dataset(dataset: str) -> None:
    if dataset not in AUPIMO_BENCHMARK_DATASETS:
        msg = f"Dataset '{dataset}' not available. Choose one of {sorted(AUPIMO_BENCHMARK_DATASETS)}."
        raise ValueError(msg)


def _get_benchmark_json_url(model: str, dataset: str) -> str:
    """Generate the URL for the JSON file of a specific model and dataset.

    Args:
        model: see `anomalib.metrics.pimo.AUPIMO_BENCHMARK_MODELS`
        dataset: "collection/category", see `anomalib.metrics.pimo.AUPIMO_BENCHMARK_DATASETS`

    Returns:
        The URL for the JSON file of the model and dataset in the benchmark from AUPIMO's official repository.
    """
    root_url = "https://raw.githubusercontent.com/jpcbertoldo/aupimo/refs/heads/main/data/experiments/benchmark"
    _validate_benchmark_model(model)
    _validate_benchmark_dataset(dataset)
    return f"{root_url}/{model}/{dataset}/aupimo/aupimos.json"


def _download_benchmark_json(url: str) -> dict:
    """Download the JSON content from an URL."""
    request = requests.get(url, timeout=10)
    return json.loads(request.text)


def aupimo_result_from_json_dict(payload: dict) -> AUPIMOResult:
    """Convert the dictionary from a JSON payload to an AUPIMOResult dataclass.

    Args:
        payload: The JSON from the benchmark results:
            {
                "fpr_lower_bound": float,
                "fpr_upper_bound": float,
                "num_thresholds": int | None,  # or "num_threshs"
                "thresh_lower_bound": float,
                "thresh_upper_bound": float,
                "aupimos": list[float],
            }

    Returns:
        An `anomalib.metrics.pimo.AUPIMOResult` dataclass.
    """
    if not isinstance(payload, dict):
        msg = f"Invalid payload. Must be a dictionary. Got {type(payload)}."
        raise TypeError(msg)
    try:
        # `num_threshs` vs `num_thresholds` is an inconsistency with an older version of the JSON file
        num_thresholds: int | None = payload["num_threshs"] if "num_threshs" in payload else payload["num_thresholds"]
        return AUPIMOResult(
            fpr_lower_bound=float(payload["fpr_lower_bound"]),
            fpr_upper_bound=float(payload["fpr_upper_bound"]),
            num_thresholds=num_thresholds if num_thresholds is None else int(num_thresholds),
            thresh_lower_bound=float(payload["thresh_lower_bound"]),
            thresh_upper_bound=float(payload["thresh_upper_bound"]),
            aupimos=torch.tensor(payload["aupimos"], dtype=torch.float64),
        )

    except KeyError as ex:
        msg = f"Invalid payload. Missing key {ex}."
        raise ValueError(msg) from ex

    except (TypeError, ValueError) as ex:
        msg = f"Invalid payload. Cause: {ex}."
        raise ValueError(msg) from ex


def aupimo_result_to_json_dict(
    aupimo_result: AUPIMOResult,
    paths: list[str | Path] | None = None,
) -> dict:
    """Convert the AUPIMOResult dataclass to a dictionary for JSON serialization.

    Args:
        aupimo_result: The AUPIMO scores from the benchmark results.
        paths: The paths of the images used to compute the AUPIMO scores. Optional.

    Returns:
        A dictionary with the AUPIMO scores and the paths.
    """
    payload = {
        "fpr_lower_bound": aupimo_result.fpr_lower_bound,
        "fpr_upper_bound": aupimo_result.fpr_upper_bound,
        "num_thresholds": aupimo_result.num_thresholds,
        "thresh_lower_bound": aupimo_result.thresh_lower_bound,
        "thresh_upper_bound": aupimo_result.thresh_upper_bound,
        "aupimos": aupimo_result.aupimos.tolist(),
    }
    if paths is not None:
        if len(paths) != aupimo_result.aupimos.shape[0]:
            msg = (
                "Invalid paths. It must have the same length as the AUPIMO scores. "
                f"Got {len(paths)} paths and {aupimo_result.aupimos.shape[0]} scores."
            )
            raise ValueError(msg)
        # make sure the paths are strings, not pathlib.Path objects
        payload["paths"] = [str(p) for p in paths]
    return payload


def _download_aupimo_benchmark_scores(model: str, dataset: str) -> tuple[dict, AUPIMOResult]:
    """Get the benchmark AUPIMO scores for a specific model and dataset from AUPIMO's official repository.

    Args:
        model: see `anomalib.metrics.pimo.AUPIMO_BENCHMARK_MODELS`
        dataset: "collection/category", see `anomalib.metrics.pimo.AUPIMO_BENCHMARK_DATASETS`

    Returns:
        (dict, AUPIMOResult): A tuple with the JSON payload and the AUPIMO scores.
            dict: The unserialized JSON from the benchmark results. See `aupimo_result_from_json_dict`.
            AUPIMOResult: The AUPIMO scores in dataclass format. See `anomalib.metrics.pimo.AUPIMOResult`.
    """
    logger.debug(f"Loading benchmark results for {model=} {dataset=}")
    url = _get_benchmark_json_url(model, dataset)
    logger.debug(f"Dowloading JSON from {url=}")
    payload = _download_benchmark_json(url)
    logger.debug("Converting json payload to dataclass")
    aupimo_result = aupimo_result_from_json_dict(payload)
    logger.debug(f"Done loading benchmark results for {model=} {dataset=}")
    return payload, aupimo_result


def _download_aupimo_benchmark_scores_multithreaded(
    model: str,
    dataset: str,
) -> tuple[tuple[str, str], tuple[dict, AUPIMOResult]]:
    """Do the same as `_download_aupimo_benchmark_scores` but return the job's arguments as well."""
    return (model, dataset), _download_aupimo_benchmark_scores(model, dataset)


def download_aupimo_benchmark_scores(
    model: str | None,
    dataset: str | None,
    avoid_multithread_download: bool = False,
) -> dict[tuple[str, str], tuple[dict, AUPIMOResult]]:
    """Dowload AUPIMO scores AUPIMO's paper benchmark (stored in the official repository).

    If `model` is None, all models are considered.
    If `dataset` is None, all datasets are considered.
    If both `model` and `dataset` are None, all combinations of models and datasets are considered.

    Official repository: https://github.com/jpcbertoldo/aupimo
    Benchmark data: https://github.com/jpcbertoldo/aupimo/tree/main/data/experiments/benchmark

    Args:
        model: The model name. Available models: `anomalib.metrics.pimo.AUPIMO_BENCHMARK_MODELS`. If None, all models.
        dataset: The "collection/category", where 'collection' is either 'mvtec' or 'visa', and 'category' is
                    the name of the dataset within the collection. Lowercase, words split by '_' (e.g. 'metal_nut').
                 Available datasets: `anomalib.metrics.pimo.AUPIMO_BENCHMARK_DATASETS`. If None, all datasets.
        avoid_multithread_download: Multi-threaded download is used by default when downloading multiple files.
                          Set this to `True` to force single-threaded download.

    Returns:
        dict[(model, dataset), (dict, AUPIMOResult)]: dictionary of results.
            key: (model, dataset) pair, e.g. ('efficientad_wr101_m_ext', 'mvtec/bottle')
            value: tuple with the JSON payload dictionary and the AUPIMO scores (dataclass).
                dict: The unserialized JSON from the benchmark results. See `aupimo_result_from_json_dict`.
                AUPIMOResult: The AUPIMO scores in dataclass format. See `anomalib.metrics.pimo.AUPIMOResult`.
    """
    if model is None:
        models = sorted(AUPIMO_BENCHMARK_MODELS)
    else:
        _validate_benchmark_model(model)
        models = [model]

    if dataset is None:
        datasets = sorted(AUPIMO_BENCHMARK_DATASETS)
    else:
        _validate_benchmark_dataset(dataset)
        datasets = [dataset]

    args = list(product(models, datasets))
    logger.debug(f"Downloading benchmark results for {len(args)} (model, dataset) pairs")

    if len(args) == 1:
        logger.debug("Using single-threaded download.")
        return {args[0]: _download_aupimo_benchmark_scores(models[0], datasets[0])}

    if avoid_multithread_download:
        logger.debug(f"Using single-threaded download due to {avoid_multithread_download=}")
        results = {}
        for model_, dataset_ in args:
            results[model_, dataset_] = _download_aupimo_benchmark_scores(model_, dataset_)
        return results

    logger.debug("Using multi-threaded download.")
    models, datasets = list(zip(*args, strict=True))  # type: ignore  # noqa: PGH003
    with ThreadPoolExecutor(thread_name_prefix="download_from_aupimo_benchmark_") as executor:
        results = executor.map(_download_aupimo_benchmark_scores_multithreaded, models, datasets)  # type: ignore  # noqa: PGH003
    return dict(results)


def get_aupimo_benchmark(
    model: str | None,
    dataset: str | None,
    avoid_multithread_download: bool = False,
) -> tuple[DataFrame, DataFrame]:
    """Dowload results from AUPIMO's paper benchmark (stored in the official repository) and format in DataFrames.

    If `model` is None, all models are considered.
    If `dataset` is None, all datasets are considered.
    If both `model` and `dataset` are None, all combinations of models and datasets are considered.

    Official repository: https://github.com/jpcbertoldo/aupimo
    Benchmark data: https://github.com/jpcbertoldo/aupimo/tree/main/data/experiments/benchmark

    Args:
        model: The model name. Available models: `anomalib.metrics.pimo.AUPIMO_BENCHMARK_MODELS`. If None, all models.
        dataset: The "collection/category", where 'collection' is either 'mvtec' or 'visa', and 'category' is
                    the name of the dataset within the collection. Lowercase, words split by '_' (e.g. 'metal_nut').
                 Available datasets: `anomalib.metrics.pimo.AUPIMO_BENCHMARK_DATASETS`. If None, all datasets.
        avoid_multithread_download: Multi-threaded download is used by default when downloading multiple files.
                          Set this to `True` to force single-threaded download.

    Returns:
        (data_per_set, data_per_image): A tuple with two DataFrames.
            data_per_set: for example, at which anomaly scores of are the integration FPRn bounds met?
            data_per_image: AUPIMO scores of each image and the path to the input image.
    """
    # don't validate model and dataset here, it's done in `download_aupimo_benchmark_scores`
    results = download_aupimo_benchmark_scores(
        model=model,
        dataset=dataset,
        avoid_multithread_download=avoid_multithread_download,
    )

    data = pd.DataFrame.from_records([
        {
            "model": model,
            "dataset": dataset,
            # per-set data
            "fpr_lower_bound": aupimo_result.fpr_lower_bound,
            "fpr_upper_bound": aupimo_result.fpr_upper_bound,
            "num_thresholds": aupimo_result.num_thresholds,
            "thresh_lower_bound": aupimo_result.thresh_lower_bound,
            "thresh_upper_bound": aupimo_result.thresh_upper_bound,
            # per-image data
            "sample_index": list(range(num_samples := len(aupimo_result.aupimos))),
            "aupimo": aupimo_result.aupimos.tolist(),
            "path": paths if (paths := json_dict["paths"]) is not None else [None] * num_samples,
        }
        for (model, dataset), (json_dict, aupimo_result) in results.items()
    ])
    data["model"] = data["model"].astype("category")
    data["dataset"] = data["dataset"].astype("category")

    data_per_set = (
        data.drop(columns=["sample_index", "aupimo", "path"]).sort_values(["model", "dataset"]).reset_index(drop=True)
    )

    data_per_image = (
        data[["model", "dataset", "sample_index", "aupimo", "path"]]
        .explode(["sample_index", "aupimo", "path"])
        .sort_values(["model", "dataset", "sample_index"])
        .reset_index(drop=True)
        .astype({"sample_index": int, "aupimo": float, "path": "string"})
    )

    return data_per_set, data_per_image
