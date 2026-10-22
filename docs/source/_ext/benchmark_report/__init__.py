"""Sphinx leaderboard and Plotly history with Python-owned CSV preparation."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import TYPE_CHECKING

from sphinx.errors import ExtensionError

from .data import load_csv, parse_benchmark_csv
from .directives import BenchmarkHistoryDirective, BenchmarkLeaderboardDirective

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.environment import BuildEnvironment


def _load_data(app: "Sphinx") -> None:
    """Fetch once per build; both directives and the download share that snapshot."""
    fixture = Path(app.srcdir) / app.config.benchmark_csv_source_fixture
    try:
        text, notice = load_csv(app.config.benchmark_csv_source, fixture)
        runs = parse_benchmark_csv(text)
    except (OSError, ValueError) as error:
        msg = f"Could not load benchmark results: {error}"
        raise ExtensionError(msg) from error
    app.env.benchmark_report_data = (text, notice, runs)
    app.env.benchmark_report_html = app.builder.format == "html"


def _updated_pages(_app: "Sphinx", env: "BuildEnvironment", *_changes: set[str]) -> list[str]:
    """Refresh live results even on an incremental documentation build."""
    return ["markdown/benchmarks"] if "markdown/benchmarks" in env.found_docs else []


def _publish_csv(app: "Sphinx", exception: Exception | None) -> None:
    """Publish the exact CSV used for charts, not an unrelated fallback fixture."""
    if exception is None and app.builder.format == "html":
        output = Path(app.outdir) / "_static/benchmark-data/results.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(app.env.benchmark_report_data[0], encoding="utf-8")


def _chart_assets(app: "Sphinx", pagename: str, *_args: object) -> None:
    """Load plotting assets only on the benchmark page."""
    if pagename == "markdown/benchmarks":
        app.add_js_file("vendor/plotly-basic-3.1.0.min.js")
        app.add_js_file("benchmark-report.js")


def setup(app: "Sphinx") -> dict:
    """Register report configuration, directives, and build hooks."""
    app.add_config_value("benchmark_csv_source", "", "env")
    app.add_config_value("benchmark_csv_source_fixture", "_static/benchmark-data/results.csv", "env")
    app.add_directive("benchmark-leaderboard", BenchmarkLeaderboardDirective)
    app.add_directive("benchmark-history", BenchmarkHistoryDirective)
    app.add_css_file("benchmark-report.css")
    app.connect("builder-inited", _load_data)
    app.connect("env-get-outdated", _updated_pages)
    app.connect("html-page-context", _chart_assets)
    app.connect("build-finished", _publish_csv)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
