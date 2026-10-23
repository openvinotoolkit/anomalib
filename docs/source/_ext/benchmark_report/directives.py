"""Docutils table builders and Sphinx directives for benchmark results."""

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from docutils import nodes  # type: ignore[import-untyped]
from sphinx.util.docutils import SphinxDirective

from .data import (
    METRICS,
    BenchmarkRun,
    LeaderboardEntry,
    category_history,
    leaderboard_for_dataset,
    sorted_versions,
)


def _load_runs(directive: SphinxDirective) -> list[BenchmarkRun]:
    directive.env.note_dependency(str(Path(directive.env.srcdir) / directive.config.benchmark_csv_source_fixture))
    return directive.env.benchmark_report_data[2]


def _build_row(cells: list[str | nodes.Node]) -> nodes.row:
    row = nodes.row()
    for cell in cells:
        entry = nodes.entry()
        paragraph = nodes.paragraph()
        paragraph += nodes.Text(cell) if isinstance(cell, str) else cell
        entry += paragraph
        row += entry
    return row


def _build_table(headers: list[str], rows: list[list[str]]) -> nodes.table:
    table = nodes.table(classes=["benchmark-table"])
    tgroup = nodes.tgroup(cols=len(headers))
    table += tgroup
    for _ in headers:
        tgroup += nodes.colspec(colwidth=1)

    thead = nodes.thead()
    tgroup += thead
    thead += _build_row(headers)

    tbody = nodes.tbody()
    tgroup += tbody
    for row in rows:
        tbody += _build_row(row)

    return table


def _format_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _model_id(dataset: str, model: str) -> str:
    return nodes.make_id(f"benchmark-{dataset}-{model}")


def _leaderboard_row(rank: int, entry: LeaderboardEntry, dataset: str) -> list:
    return [
        str(rank),
        nodes.reference("", entry.model, refuri=f"#{_model_id(dataset, entry.model)}"),
        *[_format_metric(getattr(entry, key)) for key, _, _ in METRICS],
        str(entry.category_count),
    ]


class BenchmarkLeaderboardDirective(SphinxDirective):
    """Render one ranked table per dataset for the latest anomalib version."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        """Emit a ranked table and source notice for each dataset."""
        runs = _load_runs(self)
        if not runs:
            return [nodes.paragraph(text="No benchmark results available.")]

        result: list[nodes.Node] = [
            nodes.paragraph(
                text=self.env.benchmark_report_data[1],
                classes=["benchmark-source"],
            ),
        ]
        headers = ["Rank", "Model", *[label for _, label, _ in METRICS], "Categories"]

        for dataset in sorted({run.dataset for run in runs}):
            version = sorted_versions({run.version for run in runs if run.dataset == dataset})[-1]
            entries = leaderboard_for_dataset(runs, dataset, version)
            if not entries:
                continue

            rubric = nodes.rubric(text=dataset, classes=["benchmark-dataset"])
            result.append(rubric)
            result.append(nodes.paragraph(text=f"anomalib {version}"))

            rows = [_leaderboard_row(rank, entry, dataset) for rank, entry in enumerate(entries, start=1)]
            result.append(_build_table(headers, rows))

        return result


class BenchmarkHistoryDirective(SphinxDirective):
    """Render a Plotly history chart per model, with Python-prepared data."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        """Emit model charts and category tables, with a non-HTML table fallback."""
        runs = _load_runs(self)
        result: list[nodes.Node] = []
        for dataset in sorted({run.dataset for run in runs}):
            dataset_rubric = nodes.rubric(text=dataset, classes=["benchmark-dataset"])
            result.append(dataset_rubric)

            for model in sorted({run.model for run in runs if run.dataset == dataset}):
                detail = category_history(runs, dataset, model)
                if not self.env.benchmark_report_html:
                    result.append(nodes.rubric(text=model))
                    result.append(
                        _build_table(
                            ["Run / version", *[label for _, label, _ in METRICS]],
                            [
                                [
                                    version,
                                    *[_format_metric(detail["series"][0]["values"][key][i]) for key, _, _ in METRICS],
                                ]
                                for i, version in enumerate(detail["versions"])
                            ],
                        ),
                    )
                    continue
                payload = json.dumps(detail).replace("<", "\\u003c")
                result.append(
                    nodes.raw(
                        "",
                        (
                            f'<section class="benchmark-history" id="{_model_id(dataset, model)}" tabindex="-1">'
                            f'<h4>{escape(model)} <span class="benchmark-delta">/ {escape(dataset)}</span></h4>'
                            '<div class="benchmark-controls"><label>Metric <select class="benchmark-metric">'
                            + "".join(f'<option value="{key}">{escape(label)}</option>' for key, label, _ in METRICS)
                            + "</select></label>"
                            '<label>Category <select class="benchmark-category benchmark-metric">'
                            '<option value="0">Category average</option>'
                            '<option value="all">All categories</option>'
                            + "".join(
                                f'<option value="{i}">{escape(series["label"])}</option>'
                                for i, series in enumerate(detail["series"][1:], start=1)
                            )
                            + "</select></label></div>"
                            '<div class="benchmark-chart" role="img" '
                            f'aria-label="{escape(model)} benchmark history"></div>'
                            f'<script type="application/json" class="benchmark-chart-data">{payload}</script>'
                        ),
                        format="html",
                    ),
                )
                result.append(
                    nodes.raw(
                        "",
                        (
                            '<details class="benchmark-categories"><summary>Category-wise metrics</summary>'
                            '<label>Run / version <select class="benchmark-version benchmark-metric">'
                            + "".join(
                                f'<option value="{escape(version)}"'
                                f"{' selected' if version == detail['versions'][-1] else ''}>"
                                f"{escape(version)}</option>"
                                for version in detail["versions"]
                            )
                            + "</select></label>"
                            '<p class="benchmark-delta">Click a category to inspect its history. '
                            "Click a chart point to inspect that version.</p>"
                        ),
                        format="html",
                    ),
                )
                table = _build_table(
                    ["Category", *[label for _, label, _ in METRICS]],
                    [
                        [row[0], *[_format_metric(value) for value in row[1:]]]
                        for row in detail["rows"][detail["versions"][-1]]
                    ],
                )
                table["classes"].append("benchmark-category-table")
                result.append(table)
                result.append(nodes.raw("", "</details></section>", format="html"))

        return result
