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

## Benchmark dashboard

`docs/source/markdown/benchmarks.md` embeds the interactive benchmark
dashboard via an iframe pointing at `_static/benchmarks/index.html`. That
directory is a build artifact — it is not committed, and Read the Docs
builds it automatically (see `.readthedocs.yaml`). To build it locally
before running Sphinx:

```bash
cd docs/benchmark-dashboard
npm ci
npm run build:docs
rm -rf ../source/_static/benchmarks
cp -r dist-docs ../source/_static/benchmarks
```

Without this step the benchmarks page will render an empty iframe.
