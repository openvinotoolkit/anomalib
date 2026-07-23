# Nightly Benchmark Scripts

Orchestrates a scheduled GPU benchmark against a curated model/dataset allowlist
and appends results to the Hugging Face dataset
[`anomalib/benchmarks`](https://huggingface.co/datasets/anomalib/benchmarks).

## Layout

| File                                    | Role                                                           |
| --------------------------------------- | -------------------------------------------------------------- |
| `matrix.yaml`                           | Allowlist of models/datasets and `num_categories`              |
| `orchestrate.py`                        | Select workload → run `anomalib benchmark` → append/upload CSV |
| `../../workflows/nightly-benchmark.yml` | Cron + `workflow_dispatch` on `[self-hosted, anomalib]`        |

## Secrets

| Secret     | Purpose                                                           |
| ---------- | ----------------------------------------------------------------- |
| `HF_TOKEN` | Hugging Face token with **write** access to `anomalib/benchmarks` |

Add the secret under repository Settings → Secrets and variables → Actions.

## Dataset cache

Datasets are stored under a persistent on-runner path so downloads are reused:

```text
${{ runner.tool_cache }}/anomalib/datasets
```

Override locally with `--datasets-root` or `ANOMALIB_DATASETS_ROOT`. Anomalib
downloads missing datasets automatically via each datamodule's `prepare_data()`.

## Selection rules

1. Expand `matrix.yaml` into `(model, dataset, category)` triples.
2. Prefer triples missing from HF `results.csv`.
3. Pick the `(model, dataset)` with the most remaining categories; take up to
   `num_categories`.
4. If every triple exists, pick least-recent using `run_timestamp` attached at
   upload time from the `runs/benchmark/YYYY-MM-DD-HH_MM_SS/` folder name.
5. Uniqueness for scheduling is `(model, dataset, category)` — `anomalib_version`
   is recorded by `BenchmarkJob` but does not re-queue a combo.

## Local dry-run

```bash
uv sync --extra cu130 --extra huggingface

# Print selected config only
ANOMALIB_DATASETS_ROOT=./datasets \
uv run --extra cu130 --extra huggingface \
  python .github/scripts/benchmark/orchestrate.py \
  --matrix .github/scripts/benchmark/matrix.yaml \
  --dry-run

# Run benchmark without uploading
HF_TOKEN=... \
uv run --extra cu130 --extra huggingface \
  python .github/scripts/benchmark/orchestrate.py \
  --skip-upload \
  --num-categories 1
```

## Editing the matrix

```yaml
num_categories: 5
seed: 42
accelerator: cuda
models:
  - Padim
  - name: EfficientAd
    init_args:
      model_size: small
    data_init_args:
      train_batch_size: 1
datasets:
  MVTecAD: {} # all categories
  Visa:
    categories: [candle, cashew] # optional subset
```

String model entries use defaults. Dict entries may set `init_args` and
`data_init_args` merged into the generated benchmark YAML.

## Manual workflow

Use **Actions → Nightly Benchmark → Run workflow**:

- `num_categories` — optional override
- `dry_run` — select only
- `skip_upload` — run without HF push
