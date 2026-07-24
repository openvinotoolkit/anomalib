#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Nightly benchmark orchestrator.

Selects the next (model, dataset, categories) subset from a YAML matrix, runs
``anomalib benchmark``, and appends results to the Hugging Face
``anomalib/benchmarks`` dataset.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess  # nosec B404 — invoked with a fixed, non-shell argv (see run_benchmark)
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from anomalib.data.datasets.image.mvtecad import CATEGORIES as MVTECAD_CATEGORIES
from anomalib.data.datasets.image.visa import CATEGORIES as VISA_CATEGORIES

logger = logging.getLogger(__name__)

HF_REPO_ID = "anomalib/benchmarks"
HF_REVISION = "main"
RESULTS_DIR = Path("runs") / "benchmark"
FOLDER_TS_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}_\d{2}_\d{2}$")
OLDEST_TS = "0000-00-00-00_00_00"

# Dataset class_path → (default CATEGORIES tuple, on-disk root folder name).
# CATEGORIES are imported statically to avoid dynamic ``importlib`` usage
# with runtime-computed module paths (see Semgrep non-literal-import).
DATASET_META: dict[str, tuple[tuple[str, ...], str]] = {
    "MVTecAD": (tuple(MVTECAD_CATEGORIES), "MVTecAD"),
    "Visa": (tuple(VISA_CATEGORIES), "visa"),
}

MODEL_COL = "model.class_path"
DATA_COL = "data.class_path"
CATEGORY_COL = "data.init_args.category"
TIMESTAMP_COL = "run_timestamp"


@dataclass(frozen=True)
class Triple:
    """One (model, dataset, category) combination."""

    model: str
    dataset: str
    category: str


@dataclass
class ModelSpec:
    """Model entry from the matrix allowlist."""

    name: str
    init_args: dict[str, Any] = field(default_factory=dict)
    data_init_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Selection:
    """Chosen nightly workload."""

    model: ModelSpec
    dataset: str
    categories: list[str]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(__file__).with_name("matrix.yaml"),
        help="Path to the allowlist YAML.",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=None,
        help="Persistent dataset cache root. Defaults to ANOMALIB_DATASETS_ROOT or ./datasets.",
    )
    parser.add_argument(
        "--hf-repo",
        default=HF_REPO_ID,
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--num-categories",
        type=int,
        default=None,
        help="Override matrix num_categories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select and print the workload without running benchmark or uploading.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Run the benchmark but do not upload to Hugging Face.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for generated config and intermediate files.",
    )
    return parser.parse_args()


def load_matrix(path: Path) -> dict[str, Any]:
    """Load the allowlist YAML."""
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = f"Matrix file must be a mapping: {path}"
        raise TypeError(msg)
    return data


def parse_models(raw_models: list[Any]) -> list[ModelSpec]:
    """Normalize model entries from the matrix."""
    models: list[ModelSpec] = []
    for entry in raw_models:
        if isinstance(entry, str):
            models.append(ModelSpec(name=entry))
        elif isinstance(entry, dict):
            name = entry.get("name") or entry.get("class_path")
            if not name:
                msg = f"Model entry missing name/class_path: {entry}"
                raise ValueError(msg)
            models.append(
                ModelSpec(
                    name=str(name),
                    init_args=dict(entry.get("init_args") or {}),
                    data_init_args=dict(entry.get("data_init_args") or {}),
                ),
            )
        else:
            msg = f"Unsupported model entry: {entry!r}"
            raise TypeError(msg)
    return models


def resolve_categories(dataset: str, configured: dict[str, Any] | list[str] | None) -> list[str]:
    """Resolve category list for a dataset from config or datamodule CATEGORIES."""
    if isinstance(configured, dict) and configured.get("categories"):
        return list(configured["categories"])
    if isinstance(configured, list):
        return list(configured)
    if dataset not in DATASET_META:
        msg = f"Unknown dataset {dataset!r}; add it to DATASET_META or list categories explicitly."
        raise KeyError(msg)
    categories, _ = DATASET_META[dataset]
    return list(categories)


def expand_triples(models: list[ModelSpec], datasets: dict[str, Any]) -> list[Triple]:
    """Expand the matrix into (model, dataset, category) triples."""
    triples: list[Triple] = []
    for model in models:
        for dataset_name, dataset_cfg in datasets.items():
            categories = resolve_categories(dataset_name, dataset_cfg or {})
            triples.extend(Triple(model=model.name, dataset=dataset_name, category=category) for category in categories)
    return triples


def download_hf_results(repo_id: str, dest: Path, token: str | None) -> pd.DataFrame:
    """Download existing HF results.csv, or return an empty frame if missing."""
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
    except ImportError as exc:
        msg = "huggingface_hub is required; install with `uv sync --extra huggingface`"
        raise SystemExit(msg) from exc

    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename="results.csv",
            repo_type="dataset",
            revision=HF_REVISION,
            token=token,
        )
    except (EntryNotFoundError, RepositoryNotFoundError) as exc:
        logger.warning("No existing results.csv at %s (%s); starting empty.", repo_id, exc)
        return pd.DataFrame()

    frame = pd.read_csv(local_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    return frame


def history_timestamps(history: pd.DataFrame) -> dict[Triple, str]:
    """Map each completed triple to its most recent run_timestamp."""
    if history.empty:
        return {}
    required = {MODEL_COL, DATA_COL, CATEGORY_COL}
    if not required.issubset(history.columns):
        logger.warning("History CSV missing expected columns %s; treating as empty.", required)
        return {}

    latest: dict[Triple, str] = {}
    has_ts = TIMESTAMP_COL in history.columns
    for _, row in history.iterrows():
        triple = Triple(
            model=str(row[MODEL_COL]),
            dataset=str(row[DATA_COL]),
            category=str(row[CATEGORY_COL]),
        )
        ts = str(row[TIMESTAMP_COL]) if has_ts and pd.notna(row[TIMESTAMP_COL]) else OLDEST_TS
        if triple not in latest or ts > latest[triple]:
            latest[triple] = ts
    return latest


def select_workload(
    triples: list[Triple],
    models: list[ModelSpec],
    history: pd.DataFrame,
    num_categories: int,
) -> Selection:
    """Pick one (model, dataset) and up to N categories (never-run first, else LRU)."""
    if num_categories < 1:
        msg = "num_categories must be >= 1"
        raise ValueError(msg)

    model_by_name = {spec.name: spec for spec in models}
    latest = history_timestamps(history)
    never_run = [t for t in triples if t not in latest]

    if never_run:
        by_pair: dict[tuple[str, str], list[Triple]] = defaultdict(list)
        for triple in never_run:
            by_pair[triple.model, triple.dataset].append(triple)
        pair = max(by_pair.items(), key=lambda item: (len(item[1]), item[0][0], item[0][1]))[0]
        chosen = sorted(by_pair[pair], key=lambda t: t.category)[:num_categories]
        return Selection(
            model=model_by_name[pair[0]],
            dataset=pair[1],
            categories=[t.category for t in chosen],
        )

    # All covered: choose the (model, dataset) with the oldest category timestamp,
    # then take the N least-recent categories for that pair.
    by_pair_all: dict[tuple[str, str], list[Triple]] = defaultdict(list)
    for triple in triples:
        by_pair_all[triple.model, triple.dataset].append(triple)

    def pair_oldest(pair: tuple[str, str]) -> tuple[str, str, str]:
        stamps = [latest.get(t, OLDEST_TS) for t in by_pair_all[pair]]
        return (min(stamps), pair[0], pair[1])

    pair = min(by_pair_all.keys(), key=pair_oldest)
    ordered = sorted(by_pair_all[pair], key=lambda t: (latest.get(t, OLDEST_TS), t.category))
    chosen = ordered[:num_categories]
    return Selection(
        model=model_by_name[pair[0]],
        dataset=pair[1],
        categories=[t.category for t in chosen],
    )


def dataset_root_for(datasets_root: Path, dataset: str) -> Path:
    """Return on-disk root for a dataset under the cache directory."""
    if dataset not in DATASET_META:
        return datasets_root / dataset
    return datasets_root / DATASET_META[dataset][1]


def write_benchmark_config(
    selection: Selection,
    matrix: dict[str, Any],
    datasets_root: Path,
    dest: Path,
) -> Path:
    """Write a one-shot anomalib benchmark YAML for the selection."""
    data_init: dict[str, Any] = {
        "category": {"grid": selection.categories},
        "root": str(dataset_root_for(datasets_root, selection.dataset)),
        **selection.model.data_init_args,
    }
    model_block: dict[str, Any] = {"class_path": selection.model.name}
    if selection.model.init_args:
        model_block["init_args"] = selection.model.init_args

    config = {
        "accelerator": [matrix.get("accelerator", "cuda")],
        "benchmark": {
            "seed": matrix.get("seed", 42),
            "trainer": matrix.get("trainer", {}),
            "model": model_block,
            "data": {
                "class_path": selection.dataset,
                "init_args": data_init,
            },
        },
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return dest


def run_benchmark(config_path: Path) -> None:
    """Invoke anomalib benchmark via uv."""
    cmd = ["uv", "run", "--extra", "cu130", "anomalib", "benchmark", "--config", str(config_path)]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)  # noqa: S603  # nosec B603


def find_latest_results_csv(results_root: Path = RESULTS_DIR) -> tuple[Path, str]:
    """Return the newest results.csv and its parent folder timestamp name."""
    if not results_root.is_dir():
        msg = f"No benchmark results directory at {results_root}"
        raise FileNotFoundError(msg)

    candidates = [
        (path.name, path / "results.csv")
        for path in results_root.iterdir()
        if path.is_dir() and FOLDER_TS_PATTERN.match(path.name) and (path / "results.csv").is_file()
    ]
    if not candidates:
        msg = f"No results.csv found under {results_root}"
        raise FileNotFoundError(msg)
    folder_ts, csv_path = max(candidates, key=lambda item: item[0])
    return csv_path, folder_ts


def prepare_new_rows(csv_path: Path, folder_ts: str) -> pd.DataFrame:
    """Load local results and attach run_timestamp from the results folder name."""
    frame = pd.read_csv(csv_path)
    if "anomalib_version" not in frame.columns:
        msg = f"Local results missing anomalib_version column: {csv_path}"
        raise ValueError(msg)
    frame[TIMESTAMP_COL] = folder_ts
    return frame


def append_and_upload(
    history: pd.DataFrame,
    new_rows: pd.DataFrame,
    repo_id: str,
    token: str | None,
    output_csv: Path,
) -> None:
    """Concatenate history + new rows and upload results.csv to Hugging Face."""
    combined = pd.concat([history, new_rows], ignore_index=True) if not history.empty else new_rows
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    if token is None:
        msg = "HF_TOKEN is required to upload results"
        raise SystemExit(msg)

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(output_csv),
        path_in_repo="results.csv",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Append nightly benchmark results ({datetime.now().strftime('%Y-%m-%d')})",
    )
    logger.info("Uploaded %s rows to %s (%d new).", len(combined), repo_id, len(new_rows))


def resolve_datasets_root(cli_value: Path | None) -> Path:
    """Resolve the persistent dataset cache root."""
    if cli_value is not None:
        return cli_value
    env = os.environ.get("ANOMALIB_DATASETS_ROOT")
    if env:
        return Path(env)
    return Path("datasets")


def main() -> int:
    """Orchestrate nightly selection, benchmark, and HF append."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    matrix = load_matrix(args.matrix)
    models = parse_models(matrix.get("models") or [])
    datasets = matrix.get("datasets") or {}
    if not models or not datasets:
        msg = "matrix.yaml must define non-empty models and datasets"
        raise SystemExit(msg)

    num_categories = args.num_categories if args.num_categories is not None else int(matrix.get("num_categories", 5))
    datasets_root = resolve_datasets_root(args.datasets_root)
    datasets_root.mkdir(parents=True, exist_ok=True)

    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="nightly-benchmark-"))
    work_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    history_path = work_dir / "hf_results.csv"
    history = download_hf_results(args.hf_repo, history_path, token)

    triples = expand_triples(models, datasets)
    selection = select_workload(triples, models, history, num_categories)
    logger.info(
        "Selected model=%s dataset=%s categories=%s",
        selection.model.name,
        selection.dataset,
        selection.categories,
    )

    config_path = write_benchmark_config(selection, matrix, datasets_root, work_dir / "benchmark.yaml")
    logger.info("Wrote config to %s", config_path)

    if args.dry_run:
        print(config_path.read_text(encoding="utf-8"))
        return 0

    run_benchmark(config_path)
    local_csv, folder_ts = find_latest_results_csv()
    new_rows = prepare_new_rows(local_csv, folder_ts)
    logger.info("Collected %d new rows from %s (ts=%s)", len(new_rows), local_csv, folder_ts)

    combined_path = work_dir / "results.csv"
    if args.skip_upload:
        combined = pd.concat([history, new_rows], ignore_index=True) if not history.empty else new_rows
        combined.to_csv(combined_path, index=False)
        logger.info("Wrote combined results to %s (upload skipped)", combined_path)
        return 0

    append_and_upload(history, new_rows, args.hf_repo, token, combined_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
