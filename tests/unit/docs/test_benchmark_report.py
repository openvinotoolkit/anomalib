# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline regression checks for benchmark parsing, aggregation, and Sphinx output."""

import csv
import io
import json
import re
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

import pytest
from docs.source._ext.benchmark_report import data
from sphinx.application import Sphinx


def csv_text(rows: list[dict]) -> str:
    """Produce pipeline-shaped CSV, including optional flattened configuration."""
    stream = io.StringIO()
    columns = list(dict.fromkeys([*data.REQUIRED_COLUMNS, *[column for _, _, column in data.METRICS]]))
    writer = csv.DictWriter(stream, fieldnames=[*columns, "run_timestamp", "model.layers"])
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def row(category: str = "bottle", version: str = "2.7.0", score: float = 0.9, timestamp: str = "") -> dict:
    """Build one pipeline-shaped category measurement."""
    return {
        "data.class_path": "MVTecAD",
        "data.init_args.category": category,
        "model.class_path": "Padim",
        "anomalib_version": version,
        "run_timestamp": timestamp,
        "model.layers": '["layer2", "layer3"]\nquoted config',
        **{column: score for _, _, column in data.METRICS},
    }


def test_csv_quoting() -> None:
    """BOM, CRLF, escaped quotes, embedded newlines, and unused config columns work."""
    runs = data.parse_benchmark_csv("\ufeff" + csv_text([row()]))
    assert len(runs) == 1
    assert runs[0].image_auroc == 0.9
    assert runs[0].category == "bottle"


@pytest.mark.parametrize("score", ["NaN", "inf", "", "bad", "-0.1", "1.1"])
def test_invalid_metrics(score: str) -> None:
    """Invalid measurements fail explicitly instead of becoming misleading chart values."""
    record = row()
    record["image_AUROC"] = score
    with pytest.raises(ValueError, match="Benchmark CSV record"):
        data.parse_benchmark_csv(csv_text([record]))


def test_missing_headers() -> None:
    """Renamed or missing headers fail before aggregation."""
    with pytest.raises(ValueError, match="missing required columns"):
        data.parse_benchmark_csv("data.class_path\nMVTecAD\n")


def test_absent_pixel_columns() -> None:
    """Legacy image-only CSVs need not include pixel columns at all."""
    record = {key: value for key, value in row().items() if not key.startswith("pixel_")}
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(record))
    writer.writeheader()
    writer.writerow(record)
    runs = data.parse_benchmark_csv(stream.getvalue())
    assert runs[0].pixel_f1 is None
    assert data.leaderboard_for_dataset(runs, "MVTecAD", "2.7.0")[0].pixel_auroc is None
    assert data.category_history(runs, "MVTecAD", "Padim")["series"][0]["values"]["pixel_f1"] == [None]


def test_missing_pixel_values_do_not_dilute_average() -> None:
    """Only available pixel scores contribute to the pixel average."""
    first, second = row(), row(category="grid")
    second["pixel_AUROC"] = "NaN"
    second["pixel_F1Score"] = ""
    runs = data.parse_benchmark_csv(csv_text([first, second]))
    assert data.leaderboard_for_dataset(runs, "MVTecAD", "2.7.0")[0].pixel_auroc == pytest.approx(0.9)
    assert runs[1].pixel_auroc is runs[1].pixel_f1 is None


def test_equal_category_weight_and_latest_measurements() -> None:
    """Repeated runs neither overweight categories nor replace newer measurements."""
    runs = data.parse_benchmark_csv(
        csv_text([
            row(score=0.1, timestamp="2026-10-20_01-00-00"),
            row(score=0.8, timestamp="2026-10-21_01-00-00"),
            row(score=1.0, timestamp="2026-10-21_01-00-00"),
            row(category="grid", score=0.5),
        ]),
    )
    entry = data.leaderboard_for_dataset(runs, "MVTecAD", "2.7.0")[0]
    assert entry.image_auroc == pytest.approx(0.7)
    assert entry.category_count == 2


def test_timestamped_history_and_gaps() -> None:
    """Nightly runs sharing one version remain separate; absent categories are null."""
    runs = data.parse_benchmark_csv(
        csv_text([
            row(score=0.9, timestamp="2026-10-20_01-00-00"),
            row(category="grid", score=0.6, timestamp="2026-10-20_01-00-00"),
            row(score=0.8, timestamp="2026-10-21_01-00-00"),
        ]),
    )
    history = data.category_history(runs, "MVTecAD", "Padim")
    assert len(history["versions"]) == 2
    assert history["category_counts"] == [2, 1]
    grid = next(series for series in history["series"] if series["label"] == "grid")
    assert grid["values"]["image_auroc"] == [0.6, None]
    assert history["series"][0]["values"]["image_auroc"] == [0.75, 0.8]


def test_legacy_versions() -> None:
    """Historical CSVs without timestamps retain semantic version order."""
    runs = data.parse_benchmark_csv(csv_text([row(version="2.10.0"), row(version="2.9.0")]))
    history = data.category_history(runs, "MVTecAD", "Padim")
    assert history["versions"] == ["2.9.0", "2.10.0"]


def test_hundreds_of_runs() -> None:
    """A long history retains every nightly point instead of collapsing by version."""
    runs = data.parse_benchmark_csv(
        csv_text([row(timestamp=f"2026-10-22_01-{index // 60:02d}-{index % 60:02d}") for index in range(300)]),
    )
    history = data.category_history(runs, "MVTecAD", "Padim")
    assert len(history["versions"]) == 300
    assert len(history["series"][0]["values"]["image_auroc"]) == 300


def test_offline_fallback(tmp_path: Path) -> None:
    """Network errors return a clearly identified local snapshot."""
    fixture = tmp_path / "results.csv"
    fixture.write_text(csv_text([row()]))
    with patch.object(data, "urlopen", side_effect=URLError("offline")):
        text, notice = data.load_csv("https://example.test/results.csv", fixture)
    assert text == fixture.read_text()
    assert "Offline fallback" in notice


def test_no_local_file_urls(tmp_path: Path) -> None:
    """Configured remote input accepts only HTTP(S)."""
    with pytest.raises(ValueError, match="HTTP"):
        data.load_csv("file:///etc/passwd", tmp_path / "results.csv")


def test_sphinx_build(tmp_path: Path) -> None:
    """A strict isolated Sphinx build emits links, chart payloads, and matching downloads."""
    source = tmp_path / "source"
    source.mkdir()
    extension = Path(__file__).resolve().parents[3] / "docs/source/_ext"
    (source / "conf.py").write_text(
        f"import sys\nsys.path.insert(0, {str(extension)!r})\n"
        "extensions = ['benchmark_report']\n"
        "benchmark_csv_source_fixture = 'results.csv'\n",
    )
    (source / "index.rst").write_text("Report\n======\n\n.. toctree::\n\n   markdown/benchmarks\n")
    (source / "markdown").mkdir()
    (source / "markdown/benchmarks.rst").write_text(
        "Benchmarks\n==========\n\n.. benchmark-leaderboard::\n\n.. benchmark-history::\n",
    )
    text = csv_text([row(), row(version="2.8.0")])
    (source / "results.csv").write_text(text)
    output = tmp_path / "html"
    warnings = io.StringIO()
    app = Sphinx(
        str(source),
        str(source),
        str(output),
        str(tmp_path / "doctrees"),
        "html",
        status=io.StringIO(),
        warning=warnings,
        warningiserror=True,
    )
    app.build()
    assert app.statuscode == 0, warnings.getvalue()
    html = (output / "markdown/benchmarks.html").read_text()
    assert 'href="#benchmark-mvtecad-padim"' in html
    payload = json.loads(re.search(r'class="benchmark-chart-data">(.*?)</script>', html)[1])
    assert payload["versions"] == ["2.7.0", "2.8.0"]
    assert "Bundled historical results (not live data)" in html
    assert "plotly-basic-3.1.0.min.js" in html
    assert "plotly-basic-3.1.0.min.js" not in (output / "index.html").read_text()
    assert data.parse_benchmark_csv(
        (output / "_static/benchmark-data/results.csv").read_text(),
    ) == data.parse_benchmark_csv(text)
