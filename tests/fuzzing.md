# Dataset metadata fuzzing

## Target and security property

[Atheris](https://github.com/google/atheris) runs the coverage-guided target in
`tests/fuzz/fuzz_path.py`. It exercises `resolve_path_under_root` and its validation
helpers, which guard paths read from Real-IAD JSON, Tabular metadata, and Kaputt CSV
files. Malformed metadata can contain traversal segments, absolute paths, control
characters, oversized names, or symlinks escaping the dataset directory.

Each mutated byte input is decoded with UTF-8 surrogate escaping and exercised as
a relative, dataset-rooted, and outside-rooted path, with and without extension
restrictions. Accepted paths must remain within the dataset root according to an
independent `os.path.commonpath` oracle. Valid paths must succeed; traversal,
sibling-prefix, and symlink escapes must fail. Only validation and OS path errors
are expected rejections; unexpected exceptions and failed invariants stop fuzzing.

The target instruments the production path functions and the callback explicitly,
avoiding instrumentation of unrelated ML dependencies. A temporary dataset and an
escape symlink are created once. Fuzz inputs never create or modify files.
This covers path interpretation, not CSV/JSON decoding, native memory safety, or
concurrent symlink replacement.

## Run and reproduce

Atheris requires Linux or macOS; CI uses Linux and Python 3.12. From the repository
root, install dependencies with uv:

```bash
uv sync --locked --python 3.12 --extra test --extra cpu --extra fuzz
mkdir -p fuzz-artifacts/corpus fuzz-artifacts/crashes
cp tests/fuzz/seeds/path/* fuzz-artifacts/corpus/
uv run --no-sync python tests/fuzz/fuzz_path.py fuzz-artifacts/corpus \
  -max_total_time=120 -runs=100000 -seed=3862 -max_len=1024 \
  -timeout=10 -rss_limit_mb=4096 -artifact_prefix=fuzz-artifacts/crashes/
```

CI runs on relevant pull requests, daily at 03:23 UTC, and manually. Pull requests
use seed `3862` and a 30-second budget; scheduled/manual runs use a random seed
(printed by libFuzzer) and 120 seconds. Every run is also capped at 100,000 inputs,
1,024 bytes per input, 10 seconds per callback, and a 15-minute job timeout.
Repeat with the same commit, dependencies, OS, seed, and run count; time limits can
affect how many inputs are reached. Replay a saved failure directly:

```bash
uv run --no-sync python tests/fuzz/fuzz_path.py fuzz-artifacts/crashes/crash-HASH
```

## Retain regressions

The checked-in corpus in `tests/fuzz/seeds/path/` retains traversal, absolute,
Windows, UNC, control-character, long-name, Unicode, invalid-UTF-8, and symlink
cases. Ordinary pytest runs replay every seed without requiring Atheris and check
that the oracle detects both weakened confinement and always-rejecting behavior:

```bash
uv run --no-sync pytest tests/unit/data/utils/test_path_fuzz.py
```

Windows skips UNC replay to avoid remote network access and skips the symlink test
when privileges are missing. Linux CI requires both tests to run.

CI uploads the mutated corpus, libFuzzer crash/timeout/OOM inputs, and execution
log for 30 days, including failed runs. Download useful failures before expiry,
minimize with libFuzzer, fix the boundary, and commit the minimized bytes to the
seed corpus. Checked-in seeds are replayed on every run; artifacts alone are not
permanent regression coverage. Keep generated corpora outside the source tree.

## OpenSSF Scorecard

Unlike Hypothesis ([tracking issue](https://github.com/ossf/scorecard/issues/4549)),
Atheris is recognized by the
[Scorecard Fuzzing detector](https://github.com/ossf/scorecard/blob/main/checks/raw/fuzzing.go):
it looks for `import atheris` in Python source files. The runnable harness uses
that import and executes Atheris; it is not a placeholder for detection.

This makes the Python target eligible for detection. After merging, inspect the
repository's Scorecards workflow for the actual Fuzzing result. A maximum score
is not guaranteed: Scorecard also considers other prominent repository languages.
