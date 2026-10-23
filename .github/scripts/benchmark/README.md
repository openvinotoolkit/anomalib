# Nightly benchmarks

Run a bounded subset of model/dataset/category combinations and append measured
results to [`anomalib/benchmarks`](https://huggingface.co/datasets/anomalib/benchmarks).
The dashboard consumes this CSV at docs-build time. These are internal development
runs, not guaranteed performance claims or full-paper training recipes.

## Workflow and credentials

[`nightly-benchmark.yml`](../../workflows/nightly-benchmark.yml) runs nightly at
03:00 UTC on the default branch using `[self-hosted, anomalib]`. This must be a
trusted, dedicated GPU runner, not a runner for untrusted PR code. Dependencies
come from `uv.lock` (`uv sync --frozen --extra cu130 --extra huggingface`). The
same Python interpreter runs selection and training; no nested `uv` resolution.

Configure **repository Actions secret `HF_TOKEN`** with write access scoped to
`anomalib/benchmarks`. Only the separate publication step receives it. Selection,
training, dry-runs, and skipped uploads do not need a write token. Uploads require
the default branch; manual dispatch defaults to skipping publication.

Failures remain visible in Actions. Debug artifacts are retained for 14 days;
no issues are opened automatically. The six-hour timeout is a ceiling, not a
guarantee that all hardware/model combinations will finish in that time.

## Caches and artifacts

Datasets persist at `${{ github.workspace }}/../anomalib-datasets`, configured by
`ANOMALIB_DATASETS_ROOT` (or `--datasets-root` locally). EfficientAd reuses
`imagenette` within that directory. Missing datasets are downloaded by Anomalib.

Every benchmark executes in a newly created directory under `--work-dir`. Only
that execution's output is accepted, and all selected categories must be present
exactly once. A swallowed pipeline failure, partial result, or missing output
therefore fails validation instead of publishing an earlier run.

Artifacts include the generated `benchmark.yaml`, execution logs/results,
`new_results.csv` (validated new measurements), and upload history/combined CSVs.
The workflow uses a separate work directory per run and attempt.

## Selection and training budgets

`matrix.yaml` is the allowlist. Missing triples are selected first, grouped by
model/dataset, up to `num_categories` (default five). Once all triples are covered,
the least recently measured categories rotate first. Older history without
`run_timestamp` counts as oldest. Version changes do not reset coverage.

`trainer` provides defaults; per-model `trainer` entries override them. The
checked-in matrix uses one epoch for Padim/Patchcore, at most 1,000 steps for
EfficientAd, and ten epochs for STFPM. Keep budgets fixed when comparing runs;
changing a training budget changes the experiment. Requested trainer settings,
seed, and model/data config are recorded in the flattened result columns.

```yaml
num_categories: 5
seed: 42
accelerator: cuda
models:
  - name: Stfpm
    trainer:
      max_epochs: 10
datasets:
  MVTecAD:
    categories: [bottle, cable]
  Visa: {} # all categories
trainer:
  logger: false
```

Model entries may also set `init_args` and `data_init_args` (e.g. EfficientAd needs
`train_batch_size: 1`). Dataset root/category selection and benchmark-owned trainer
arguments (`accelerator`, `devices`, `default_root_dir`) must not be overridden.
Leave checkpointing enabled: Engine installs a checkpoint callback. The benchmark
job's temporary directory contains checkpoints and is cleaned up after the job.
Bad matrix types, duplicate model/category names, and unknown categories fail early.

## CSV contract and publication

Required columns: `model.class_path`, `data.class_path`, `data.init_args.category`,
`anomalib_version`, `seed`, `image_AUROC`, `image_F1Score`. Scores must be finite
and between zero and one. Image-only models may omit `pixel_AUROC` and
`pixel_F1Score`, or leave them empty; the dashboard displays gaps/dashes.

New measurements add `run_timestamp` from the result folder name
(`YYYY-MM-DD-HH_MM_SS`), `source_commit` (`GITHUB_SHA`) and `workflow_run_id`
(`GITHUB_RUN_ID`), when available. Historical rows need not have these fields.

The full CSV is intentionally append-only, without a rolling retention limit.
Publication re-downloads history at a resolved immutable HF commit, validates
both inputs, and uploads with `parent_commit` set to that revision. Authentication,
network, corrupt history, and concurrent-write errors fail closed. Only an absent
`results.csv` in an accessible repository initializes empty history.

The measurement identity is `(model, dataset, category, run_timestamp, seed)`.
Retrying already-published identities is a no-op; distinct timestamps remain
separate observations. Resolve a concurrent-update failure by retrying publication
of the saved `new_results.csv`; it is not necessary to rerun training.

## Local validation and manual runs

```bash
uv sync --frozen --extra cu130 --extra huggingface

# Select and print a config; no training or write secret.
uv run --frozen --no-sync python .github/scripts/benchmark/orchestrate.py \
  --dry-run --work-dir ./results/nightly-preview

# Run one selected category and stage output; no upload or write secret.
uv run --frozen --no-sync python .github/scripts/benchmark/orchestrate.py \
  --num-categories 1 --skip-upload --work-dir ./results/nightly-preview

# Separate publication command, only when intentionally publishing measured data.
# HF_TOKEN must be set in the environment for this command.
uv run --frozen --no-sync python .github/scripts/benchmark/orchestrate.py \
  --upload-results ./results/nightly-preview/new_results.csv \
  --work-dir ./results/nightly-preview
```

In **Actions → Nightly Benchmark → Run workflow**, choose `dry_run` or leave
`skip_upload` enabled for validation. Preview-branch dispatches never publish.

Offline scheduler tests need no GPU, dataset downloads, or credentials:

```bash
uv run --no-project --with pytest --with pandas --with pyyaml --with huggingface_hub \
  python -m pytest --confcutdir=tests/unit/pipelines tests/unit/pipelines/test_nightly_benchmark.py
```
