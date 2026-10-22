"""Parse nightly benchmark CSVs and prepare leaderboard/category histories."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import io
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from statistics import mean
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from pathlib import Path

# Change these labels/fields to add or remove a displayed metric.
METRICS = (
    ("image_auroc", "Image AUROC", "image_AUROC"),
    ("pixel_auroc", "Pixel AUROC", "pixel_AUROC"),
    ("image_f1", "Image F1", "image_F1Score"),
    ("pixel_f1", "Pixel F1", "pixel_F1Score"),
)
REQUIRED_COLUMNS = (
    "data.class_path",
    "data.init_args.category",
    "model.class_path",
    "anomalib_version",
    *(column for _, _, column in METRICS),
)


@dataclass(frozen=True)
class BenchmarkRun:
    """One measured category result; historical CSVs may have no timestamp."""

    dataset: str
    category: str
    model: str
    version: str
    image_auroc: float
    pixel_auroc: float
    image_f1: float
    pixel_f1: float
    timestamp: str = ""


@dataclass(frozen=True)
class LeaderboardEntry:
    """Category-averaged metrics for a model and version/run."""

    model: str
    version: str
    category_count: int
    image_auroc: float
    pixel_auroc: float
    image_f1: float
    pixel_f1: float
    timestamp: str = ""

    @property
    def label(self) -> str:
        """Identify timestamped nightly runs without merging identical versions."""
        return f"{self.version} · {self.timestamp}" if self.timestamp else self.version


def _parse_row(row: dict) -> BenchmarkRun:
    if None in row or any(not row[column] for column in REQUIRED_COLUMNS):
        msg = "missing or extra fields"
        raise ValueError(msg)
    metrics = {key: float(row[column]) for key, _, column in METRICS}
    if not all(isfinite(value) and 0 <= value <= 1 for value in metrics.values()):
        msg = "metrics must be finite scores between 0 and 1"
        raise ValueError(msg)
    return BenchmarkRun(
        dataset=row["data.class_path"],
        category=row["data.init_args.category"],
        model=row["model.class_path"],
        version=row["anomalib_version"],
        timestamp=row.get("run_timestamp") or "",
        **metrics,
    )


def parse_benchmark_csv(text: str) -> list[BenchmarkRun]:
    """Validate required columns and finite metric values using standard CSV quoting."""
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")), strict=True)
    missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
    if missing:
        msg = f"Benchmark CSV is missing required columns: {', '.join(sorted(missing))}"
        raise ValueError(msg)
    runs = []
    try:
        runs = [_parse_row(row) for row in reader]
    except (TypeError, ValueError, csv.Error) as error:
        msg = f"Benchmark CSV record ending at line {reader.line_num}: {error}"
        raise ValueError(msg) from error
    return runs


def load_csv(source: str, fixture: Path) -> tuple[str, str]:
    """Fetch configured HTTP(S) data or return the fixture with a visible source notice."""
    if source:
        if urlsplit(source).scheme not in {"http", "https"}:
            msg = "benchmark_csv_source must be an HTTP(S) URL"
            raise ValueError(msg)
        try:
            # URL is trusted docs configuration with an explicitly validated scheme.
            with urlopen(source, timeout=15) as response:  # noqa: S310 # nosec B310 - validated HTTP(S) config
                return response.read().decode("utf-8-sig"), "Published benchmark results"
        except (URLError, TimeoutError, OSError):
            return (
                fixture.read_text(encoding="utf-8-sig"),
                "Offline fallback: bundled historical results (not live data)",
            )
    return fixture.read_text(encoding="utf-8-sig"), "Bundled historical results (not live data)"


def sorted_versions(versions: set[str]) -> list[str]:
    """Order versions semantically, with unknown version strings first."""

    def key(value: str) -> tuple[Version, str]:
        try:
            return Version(value), value
        except InvalidVersion:
            return Version("0.dev0"), value

    return sorted(versions, key=key)


def _entry(model: str, version: str, runs: list[BenchmarkRun], timestamp: str = "") -> LeaderboardEntry:
    # Repeated measurements are averaged within a category first, giving each
    # category equal weight instead of overweighting frequently measured ones.
    categories: dict[str, list[BenchmarkRun]] = defaultdict(list)
    for run in runs:
        categories[run.category].append(run)
    return LeaderboardEntry(
        model=model,
        version=version,
        timestamp=timestamp,
        category_count=len(categories),
        **{
            key: mean(mean(getattr(run, key) for run in group) for group in categories.values())
            for key, _, _ in METRICS
        },
    )


def leaderboard_for_dataset(runs: list[BenchmarkRun], dataset: str, version: str) -> list[LeaderboardEntry]:
    """Rank models using each category's latest measurements within the selected version."""
    selected = [run for run in runs if run.dataset == dataset and run.version == version]
    by_category: dict[tuple[str, str], list[BenchmarkRun]] = defaultdict(list)
    for run in selected:
        by_category[run.model, run.category].append(run)
    by_model: dict[str, list[BenchmarkRun]] = defaultdict(list)
    for (model, _), group in by_category.items():
        latest = max(run.timestamp for run in group)
        by_model[model].extend(run for run in group if run.timestamp == latest)
    entries = [_entry(model, version, group) for model, group in by_model.items()]
    return sorted(entries, key=lambda entry: (-entry.image_auroc, entry.model))


def category_history(runs: list[BenchmarkRun], dataset: str, model: str) -> dict:
    """Prepare run-aligned chart series and category rows, preserving missing values as gaps."""
    selected = [run for run in runs if run.dataset == dataset and run.model == model]
    versions = sorted_versions({run.version for run in selected})
    groups: dict[tuple[str, str], list[BenchmarkRun]] = defaultdict(list)
    for run in selected:
        groups[run.timestamp, run.version].append(run)
    # Legacy rows precede timestamped nightly runs. Pipeline timestamps are
    # sortable folder timestamps (YYYY-MM-DD_HH-MM-SS).
    keys = sorted(groups, key=lambda pair: (pair[0], versions.index(pair[1])))
    entries = [_entry(model, version, groups[timestamp, version], timestamp) for timestamp, version in keys]
    labels = [entry.label for entry in entries]
    series = [
        {
            "label": "Category average",
            "values": {key: [getattr(entry, key) for entry in entries] for key, _, _ in METRICS},
        },
    ]
    rows: dict[str, list] = {label: [] for label in labels}
    for category in sorted({run.category for run in selected}):
        category_entries = []
        for label, (timestamp, version) in zip(labels, keys, strict=True):
            group = [run for run in groups[timestamp, version] if run.category == category]
            entry = _entry(model, version, group, timestamp) if group else None
            category_entries.append(entry)
            if entry:
                rows[label].append([category, *[getattr(entry, key) for key, _, _ in METRICS]])
        series.append({
            "label": category,
            "values": {
                key: [getattr(entry, key) if entry else None for entry in category_entries] for key, _, _ in METRICS
            },
        })
    return {
        "versions": labels,
        "series": series,
        "rows": rows,
        "category_counts": [entry.category_count for entry in entries],
        "metrics": [{"key": key, "label": label} for key, label, _ in METRICS],
    }
