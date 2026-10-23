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
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError

logger = logging.getLogger(__name__)

HF_REPO_ID = "anomalib/benchmarks"
RESULTS_DIR = Path("runs") / "benchmark"
FOLDER_TS_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}_\d{2}_\d{2}$")
OLDEST_TS = "0000-00-00-00_00_00"

DATASET_ROOTS = {"MVTecAD": "MVTecAD", "Visa": "visa"}

MODEL_COL = "model.class_path"
DATA_COL = "data.class_path"
CATEGORY_COL = "data.init_args.category"
TIMESTAMP_COL = "run_timestamp"
REQUIRED_COLUMNS = (MODEL_COL, DATA_COL, CATEGORY_COL, "anomalib_version", "seed", "image_AUROC", "image_F1Score")
METRIC_COLUMNS = ("image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score")


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
    trainer: dict[str, Any] = field(default_factory=dict)


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
        "--upload-results",
        type=Path,
        help="Validate and append an already-completed run. Only this mode requires HF_TOKEN.",
    )
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
    if not isinstance(data.get("models"), list) or not data["models"]:
        msg = "matrix.models must be a non-empty list"
        raise ValueError(msg)
    if not isinstance(data.get("datasets"), dict) or not data["datasets"]:
        msg = "matrix.datasets must be a non-empty mapping"
        raise ValueError(msg)
    if type(data.get("num_categories", 5)) is not int or data.get("num_categories", 5) < 1:
        msg = "matrix.num_categories must be a positive integer"
        raise ValueError(msg)
    if not isinstance(data.get("trainer", {}), dict):
        msg = "matrix.trainer must be a mapping"
        raise TypeError(msg)
    if type(data.get("seed", 42)) is not int:
        msg = "matrix.seed must be an integer"
        raise ValueError(msg)
    if data.get("accelerator", "cuda") not in {"cpu", "cuda"}:
        msg = "matrix.accelerator must be cpu or cuda"
        raise ValueError(msg)
    return data


def parse_models(raw_models: list[Any]) -> list[ModelSpec]:
    """Normalize model entries from the matrix."""
    models: list[ModelSpec] = []
    if not isinstance(raw_models, list):
        msg = "models must be a list"
        raise TypeError(msg)
    for entry in raw_models:
        if isinstance(entry, str):
            models.append(ModelSpec(name=entry))
        elif isinstance(entry, dict):
            name = entry.get("name") or entry.get("class_path")
            if not name:
                msg = f"Model entry missing name/class_path: {entry}"
                raise ValueError(msg)
            for key in ("init_args", "data_init_args", "trainer"):
                if not isinstance(entry.get(key, {}), dict):
                    msg = f"Model {name}: {key} must be a mapping"
                    raise TypeError(msg)
            if {"root", "category"} & entry.get("data_init_args", {}).keys():
                msg = f"Model {name}: root/category are controlled by the workload selection"
                raise ValueError(msg)
            models.append(
                ModelSpec(
                    name=str(name),
                    init_args=dict(entry.get("init_args") or {}),
                    data_init_args=dict(entry.get("data_init_args") or {}),
                    trainer=dict(entry.get("trainer") or {}),
                ),
            )
        else:
            msg = f"Unsupported model entry: {entry!r}"
            raise TypeError(msg)
    names = [model.name for model in models]
    if len(names) != len(set(names)) or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) for name in names):
        msg = "Model names must be unique, simple anomalib model class names"
        raise ValueError(msg)
    return models


def default_categories(dataset: str) -> list[str]:
    """Resolve the explicit dataset allowlist without loading models for upload-only runs."""
    if dataset == "MVTecAD":
        from anomalib.data.datasets.image.mvtecad import CATEGORIES as MVTECAD_CATEGORIES

        return list(MVTECAD_CATEGORIES)
    if dataset == "Visa":
        from anomalib.data.datasets.image.visa import CATEGORIES as VISA_CATEGORIES

        return list(VISA_CATEGORIES)
    msg = f"Unknown dataset {dataset!r}; supported datasets: {', '.join(DATASET_ROOTS)}"
    raise ValueError(msg)


def resolve_categories(dataset: str, configured: dict[str, Any] | list[str] | None) -> list[str]:
    """Resolve category list for a dataset from config or datamodule CATEGORIES."""
    allowed = default_categories(dataset)
    if configured is None or configured == {}:
        return allowed
    categories = configured.get("categories") if isinstance(configured, dict) else configured
    if not isinstance(categories, list) or not categories or any(not isinstance(c, str) for c in categories):
        msg = f"Dataset {dataset}: categories must be a non-empty list of names"
        raise ValueError(msg)
    if len(categories) != len(set(categories)) or set(categories) - set(allowed):
        msg = f"Dataset {dataset}: duplicate or unknown categories: {categories}"
        raise ValueError(msg)
    return categories


def expand_triples(models: list[ModelSpec], datasets: dict[str, Any]) -> list[Triple]:
    """Expand the matrix into (model, dataset, category) triples."""
    triples: list[Triple] = []
    for model in models:
        for dataset_name, dataset_cfg in datasets.items():
            categories = resolve_categories(dataset_name, dataset_cfg)
            triples.extend(Triple(model=model.name, dataset=dataset_name, category=category) for category in categories)
    return triples


def download_hf_results(repo_id: str, dest: Path, token: str | None = None) -> tuple[pd.DataFrame, str]:
    """Read one immutable HF snapshot; only a missing CSV means empty history.

    Resolve the current data revision to a commit SHA before downloading. Auth,
    repository, and network errors propagate rather than risking history loss.
    The returned SHA is also used as the upload's optimistic concurrency guard.
    """
    revision = HfApi(token=token or False).dataset_info(repo_id=repo_id).sha
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        msg = "Hugging Face did not return an immutable dataset commit SHA"
        raise ValueError(msg)
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename="results.csv",
            repo_type="dataset",
            revision=revision,
            token=token or False,
        )
    except RemoteEntryNotFoundError as exc:
        logger.warning("No existing results.csv at %s (%s); starting empty.", repo_id, exc)
        return pd.DataFrame(), revision

    frame = pd.read_csv(local_path)
    validate_results(frame)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    return frame, revision


def validate_results(frame: pd.DataFrame, *, timestamp_required: bool = False) -> None:
    """Reject broken schemas and scores; image-only models may omit pixel scores."""
    required = (*REQUIRED_COLUMNS, TIMESTAMP_COL) if timestamp_required else REQUIRED_COLUMNS
    missing = set(required) - set(frame.columns)
    if missing or frame.empty:
        msg = f"Empty or invalid benchmark CSV; missing columns: {sorted(missing)}"
        raise ValueError(msg)
    if frame[list(required)].isna().any().any() or frame[list(required)].astype(str).eq("").any().any():
        msg = "Required benchmark fields must not be empty"
        raise ValueError(msg)
    if timestamp_required and not frame[TIMESTAMP_COL].astype(str).str.fullmatch(FOLDER_TS_PATTERN).all():
        msg = "Invalid run_timestamp; expected benchmark folder timestamp"
        raise ValueError(msg)
    for column in METRIC_COLUMNS:
        if column not in frame:
            continue
        scores = pd.to_numeric(frame[column].dropna(), errors="raise")
        if not scores.map(isfinite).all() or not scores.between(0, 1).all():
            msg = f"{column} must contain finite scores between 0 and 1"
            raise ValueError(msg)


def history_timestamps(history: pd.DataFrame) -> dict[Triple, str]:
    """Map each completed triple to its most recent run_timestamp."""
    if history.empty:
        return {}
    required = {MODEL_COL, DATA_COL, CATEGORY_COL}
    if not required.issubset(history.columns):
        msg = f"History CSV missing scheduling columns: {sorted(required - set(history.columns))}"
        raise ValueError(msg)

    latest: dict[Triple, str] = {}
    has_ts = TIMESTAMP_COL in history.columns
    for _, row in history.iterrows():
        triple = Triple(
            model=str(row[MODEL_COL]),
            dataset=str(row[DATA_COL]),
            category=str(row[CATEGORY_COL]),
        )
        ts = str(row[TIMESTAMP_COL]) if has_ts and pd.notna(row[TIMESTAMP_COL]) else OLDEST_TS
        if ts != OLDEST_TS and not FOLDER_TS_PATTERN.fullmatch(ts):
            msg = f"Invalid historical run_timestamp: {ts}"
            raise ValueError(msg)
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
    if not triples:
        msg = "The benchmark matrix resolves to no workloads"
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
    return datasets_root / DATASET_ROOTS[dataset]


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
        model_block["init_args"] = dict(selection.model.init_args)
    if selection.model.name == "EfficientAd":
        model_block.setdefault("init_args", {}).setdefault("imagenet_dir", str(datasets_root / "imagenette"))

    config = {
        "accelerator": [matrix.get("accelerator", "cuda")],
        "benchmark": {
            "seed": matrix.get("seed", 42),
            "trainer": {**matrix.get("trainer", {}), **selection.model.trainer},
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


def run_benchmark(config_path: Path, run_dir: Path) -> None:
    """Run the pipeline in the current interpreter, isolated from previous outputs."""
    script = (
        "from argparse import Namespace; import sys; "
        "from anomalib.pipelines import Benchmark; "
        "Benchmark().run(Namespace(config=sys.argv[1]))"
    )
    cmd = [sys.executable, "-c", script, str(config_path.resolve())]
    logger.info("Running: %s", " ".join(cmd))
    # Inherited write credentials must not reach training code, even in local runs.
    env = {**os.environ, "HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": ""}
    subprocess.run(cmd, check=True, cwd=run_dir, env=env)  # noqa: S603  # nosec B603


def find_latest_results_csv(results_root: Path = RESULTS_DIR) -> tuple[Path, str]:
    """Require exactly one output inside this execution's fresh results directory."""
    if not results_root.is_dir():
        msg = f"No benchmark results directory at {results_root}"
        raise FileNotFoundError(msg)

    candidates = [
        (path.name, path / "results.csv")
        for path in results_root.iterdir()
        if path.is_dir() and FOLDER_TS_PATTERN.match(path.name) and (path / "results.csv").is_file()
    ]
    if len(candidates) != 1:
        msg = f"Expected exactly one fresh results.csv under {results_root}; found {len(candidates)}"
        raise FileNotFoundError(msg)
    folder_ts, csv_path = max(candidates, key=lambda item: item[0])
    return csv_path, folder_ts


def prepare_new_rows(csv_path: Path, folder_ts: str, selection: Selection) -> pd.DataFrame:
    """Validate complete workload output before adding provenance and publication metadata."""
    frame = pd.read_csv(csv_path)
    validate_results(frame)
    expected = {(selection.model.name, selection.dataset, category) for category in selection.categories}
    actual = set(frame[[MODEL_COL, DATA_COL, CATEGORY_COL]].itertuples(index=False, name=None))
    if actual != expected or len(frame) != len(expected):
        msg = f"Benchmark output does not match selected workload: expected {expected}, got {actual}"
        raise ValueError(msg)
    frame[TIMESTAMP_COL] = folder_ts
    frame["source_commit"] = os.environ.get("GITHUB_SHA", "")
    frame["workflow_run_id"] = os.environ.get("GITHUB_RUN_ID", "")
    validate_results(frame, timestamp_required=True)
    return frame


def append_and_upload(
    new_rows: pd.DataFrame,
    repo_id: str,
    token: str | None,
    output_csv: Path,
) -> None:
    """Append idempotently to fresh history, rejecting concurrent dataset changes."""
    if not token:
        msg = "HF_TOKEN is required to upload results"
        raise SystemExit(msg)

    validate_results(new_rows, timestamp_required=True)
    history, revision = download_hf_results(repo_id, output_csv.with_name("upload_history.csv"), token)
    keys = [MODEL_COL, DATA_COL, CATEGORY_COL, TIMESTAMP_COL, "seed"]
    if new_rows.duplicated(keys).any():
        msg = "New results contain duplicate measurement identities"
        raise ValueError(msg)
    if not history.empty and TIMESTAMP_COL in history:
        existing = pd.MultiIndex.from_frame(history[keys])
        new_rows = new_rows.loc[~pd.MultiIndex.from_frame(new_rows[keys]).isin(existing)]
    combined = pd.concat([history, new_rows], ignore_index=True) if not history.empty else new_rows
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    if new_rows.empty:
        logger.info("Measurements already published; no upload needed.")
        return
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(output_csv),
        path_in_repo="results.csv",
        repo_id=repo_id,
        repo_type="dataset",
        parent_commit=revision,
        commit_message=f"Append nightly benchmark results ({datetime.now(timezone.utc).strftime('%Y-%m-%d')})",
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

    work_dir = (args.work_dir or Path(tempfile.mkdtemp(prefix="nightly-benchmark-"))).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.upload_results:
        if args.dry_run or args.skip_upload:
            msg = "--upload-results cannot be combined with --dry-run or --skip-upload"
            raise ValueError(msg)
        append_and_upload(
            pd.read_csv(args.upload_results),
            args.hf_repo,
            os.environ.get("HF_TOKEN"),
            work_dir / "results.csv",
        )
        return 0

    matrix = load_matrix(args.matrix)
    models = parse_models(matrix.get("models") or [])
    datasets = matrix.get("datasets") or {}
    if not models or not datasets:
        msg = "matrix.yaml must define non-empty models and datasets"
        raise SystemExit(msg)

    num_categories = args.num_categories if args.num_categories is not None else int(matrix.get("num_categories", 5))
    datasets_root = resolve_datasets_root(args.datasets_root).resolve()
    datasets_root.mkdir(parents=True, exist_ok=True)

    history_path = work_dir / "hf_results.csv"
    history, _ = download_hf_results(args.hf_repo, history_path)

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

    run_dir = Path(tempfile.mkdtemp(prefix="execution-", dir=work_dir))
    run_benchmark(config_path, run_dir)
    local_csv, folder_ts = find_latest_results_csv(run_dir / RESULTS_DIR)
    new_rows = prepare_new_rows(local_csv, folder_ts, selection)
    new_rows.to_csv(work_dir / "new_results.csv", index=False)
    logger.info("Collected %d new rows from %s (ts=%s)", len(new_rows), local_csv, folder_ts)

    combined_path = work_dir / "results.csv"
    if args.skip_upload:
        combined = pd.concat([history, new_rows], ignore_index=True) if not history.empty else new_rows
        combined.to_csv(combined_path, index=False)
        logger.info("Wrote combined results to %s (upload skipped)", combined_path)
        return 0

    logger.info("Validated results at %s. Publish using --upload-results.", work_dir / "new_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
