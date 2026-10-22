# Anomalib Documentation

This directory contains the documentation source for Anomalib, built using Sphinx, Sphinx Book Theme, and MyST Parser.

## Installation

Install documentation dependencies using `uv` from the repository root:

```bash
uv sync --extra docs
```

For a development environment with all extras (models, tests, docs):

```bash
uv sync --extra dev
```

## Building Documentation

To build the HTML documentation:

```bash
uv run sphinx-build -b html docs/source docs/build/html
```

To run a strict build where warnings are treated as errors:

```bash
uv run sphinx-build -b html -W --keep-going docs/source docs/build/html
```

To check for broken external links:

```bash
uv run sphinx-build -b linkcheck docs/source docs/build/linkcheck
```

## Structure

- `docs/source/index.md`: Main documentation landing page and root toctree.
- `docs/source/markdown/get_started/`: Quickstart tutorials and migration guides.
- `docs/source/markdown/guides/how_to/`: Goal-oriented how-to guides.
- `docs/source/markdown/guides/reference/`: API, CLI, model, and datamodule references.
- `docs/source/markdown/guides/developer/`: Contributor guidelines and architecture design.
- `docs/source/examples`: Symlink to `examples/` at the repository root.

## Benchmark report

The benchmark page uses the existing Sphinx theme with a locally served Plotly.js
basic bundle. No Node installation or frontend build is required.

- `_ext/benchmark_report/data.py`: CSV parsing, equal-category averages, and history.
- `_ext/benchmark_report/directives.py`: leaderboard and model/category panels.
- `_static/benchmark-report.js` and `.css`: chart interaction and presentation.

These paths are relative to `docs/source/`. To change a metric, edit `METRICS` in
`data.py`; it supplies chart controls and category-table labels.

Local builds use bundled historical measurements. Read the Docs fetches the
published CSV once per build. A network failure uses the bundled measurements
with a visible fallback notice; invalid CSV fails the build. The download always
contains the same CSV used for the report. Fresh nightly data appears after the
next docs build, not on every page visit.

To build against the published data locally:

```bash
uv run sphinx-build -b html docs/source docs/build/html \
  -D benchmark_csv_source=https://huggingface.co/datasets/anomalib/benchmarks/resolve/main/results.csv
```

History preserves `run_timestamp` from the nightly pipeline. Older CSVs without
timestamps use versions instead. Each history point averages only that run's
available categories; changing coverage is not automatically a regression.

Run the focused, offline tests without loading the model/data fixtures:

```bash
uv run --extra docs --extra test python -m pytest --confcutdir=tests/unit/docs tests/unit/docs/test_benchmark_report.py
```

Plotly basic v3.1.0 is pinned under `_static/vendor/` with its MIT license. When
updating, use the upstream basic distribution, preserve the license and inventory
entry, and update the narrow pre-commit exemptions for that minified vendor file.
