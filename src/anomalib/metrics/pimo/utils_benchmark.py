"""Utility functions to compare AUPIMO scores with the benchmark results from AUPIMO's official repository."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import urllib.request
from pathlib import Path

import torch

from .dataclasses import AUPIMOResult

logger = logging.getLogger(__name__)


def _get_benchmark_scores_url(model: str, dataset: str) -> str:
    """Generate the URL for the JSON file of a specific model and dataset.

    Args:
        model: The model name. See `_get_json_url` for the available models.
               Available models: https://github.com/jpcbertoldo/aupimo/tree/main/data/experiments/benchmark
        dataset: The "collection/dataset", where 'collection' is either 'mvtec' or 'visa', and 'dataset' is
                    the name of the dataset within the collection (lowercase, words split by '_').
                Available datasets:
                https://github.com/jpcbertoldo/aupimo/tree/main/data/experiments/benchmark/efficientad_wr101_m_ext
    Returns:
        The URL for the JSON file of the model and dataset in the benchmark from AUPIMO's official repository.
        Reference: https://github.com/jpcbertoldo/aupimo
    """
    root_url = "https://raw.githubusercontent.com/jpcbertoldo/aupimo/refs/heads/main/data/experiments/benchmark"
    models = {
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
    if model not in models:
        msg = f"Model '{model}' not available. Choose one of {sorted(models)}."
        raise ValueError(msg)
    datasets = {
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
    if dataset not in datasets:
        msg = f"Dataset '{dataset}' not available. Choose one of {sorted(datasets)}."
        raise ValueError(msg)
    return f"{root_url}/{model}/{dataset}/aupimo/aupimos.json"


def _download_json(url_str: str) -> dict[str, str | float | int | list[str]]:
    """Download the JSON content from an URL."""
    with urllib.request.urlopen(url_str) as url:  # noqa: S310
        return json.load(url)


def load_aupimo_result_from_json_dict(payload: dict) -> AUPIMOResult:
    """Convert the JSON payload to an AUPIMOResult dataclass."""
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


def get_benchmark_aupimo_scores(
    model: str,
    dataset: str,
    verbose: bool = True,
) -> tuple[dict[str, str | float | int | list[str]], AUPIMOResult]:
    """Get the benchmark AUPIMO scores for a specific model and dataset.

    Args:
        model: The model name. See `_get_json_url` for the available models.
        dataset: The "collection/dataset", where 'collection' is either 'mvtec' or 'visa', and 'dataset' is
                 the name of the dataset within the collection. See `_get_json_url` for the available datasets.
        verbose: Whether to logger.debug the progress.

    Returns:
        A `AUPIMOResult` dataclass with the AUPIMO scores from the benchmark results.

        More details in our paper: https://arxiv.org/abs/2401.01984
    """
    if verbose:
        logger.debug(f"Loading benchmark results for model '{model}' and dataset '{dataset}'")
    url = _get_benchmark_scores_url(model, dataset)
    if verbose:
        logger.debug(f"Dowloading JSON file from {url}")
    payload = _download_json(url)
    if verbose:
        logger.debug("Converting payload to dataclass")
    aupimo_result = load_aupimo_result_from_json_dict(payload)
    if verbose:
        logger.debug("Done!")
    return payload, aupimo_result


def save_aupimo_result_to_json_dict(
    aupimo_result: AUPIMOResult,
    paths: list[str | Path] | None = None,
) -> dict[str, str | float | int | list[str]]:
    """Convert the AUPIMOResult dataclass to a JSON payload."""
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
