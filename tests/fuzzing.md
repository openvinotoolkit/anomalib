# Dataset metadata fuzzing

## Target and security property

`tests/unit/data/utils/test_path_fuzz.py` fuzzes `resolve_path_under_root` and its
`validate_path` boundary. Real-IAD JSON, Tabular metadata, and Kaputt CSV readers
use this boundary before accessing image or mask paths. Untrusted metadata can
contain traversal segments, absolute paths, control characters, oversized names,
or paths through symlinks that escape the dataset directory.

Hypothesis generates arbitrary Unicode strings and structured path components.
Accepted paths must remain within the dataset root, checked with an independent
`os.path.commonpath` oracle, and retain the permitted extension. Separate generated
tests require valid paths to succeed and sibling-prefix, traversal, and symlink
escapes to fail. Inputs are never used to write files. Only validation errors and
OS path errors are accepted rejections; unexpected exceptions fail the test.

This targets the path interpretation boundary, not the CSV/JSON decoder or archive
decompression. It does not simulate concurrent symlink replacement. On Windows,
UNC paths are excluded to avoid contacting remote hosts during local tests.

## Run and reproduce

Install the test dependencies with uv, then run from the repository root:

```bash
uv sync --locked --python 3.12 --extra test --extra cpu
uv run --no-sync pytest tests/unit/data/utils/test_path_fuzz.py \
  --hypothesis-seed=3862 --hypothesis-show-statistics --timeout=120
```

Each of the three properties runs at most 1,000 generated examples, plus explicit
regression examples and failure shrinking. Each test has a 120-second timeout in
CI; the complete job has a 15-minute timeout. The tests also run with the ordinary
unit suite. Windows without symlink privileges skips only the symlink property;
the Linux CI job requires that property to run.

The dedicated workflow runs on relevant pull requests, daily at 03:23 UTC, and
manually. Pull requests use seed `3862`; scheduled and manual runs use the GitHub
run ID to explore different inputs. The workflow prints the seed. Replay with the
same commit, locked dependencies, Python version, OS, and seed. Hypothesis also
prints a temporary `@reproduce_failure` decorator for a minimized failure.

## Retain regressions

The workflow retains JUnit failure details and any Hypothesis example database for
30 days, including failed runs. A fixed seed can disable the database; the JUnit
report and printed reproduction blob still preserve the failure. Download the
artifact before it expires, reproduce the failure, fix the boundary, and add its
minimized input as an `@example` or a focused regression test in this repository.
Checked-in examples run on every execution and survive dependency upgrades;
opaque reproduction blobs and the transient database do not replace them.

## OpenSSF Scorecard gap

This integration does not claim a passing Scorecard Fuzzing check. The
[Scorecard Fuzzing documentation](https://github.com/ossf/scorecard/blob/main/docs/checks.md#fuzzing)
and [detection implementation](https://github.com/ossf/scorecard/blob/main/checks/raw/fuzzing.go)
were reviewed on 2026-10-22. They detect OSS-Fuzz, ClusterFuzzLite, and selected
language-specific fuzzing libraries; standalone Python Hypothesis tests are not
detected. The repository pins `ossf/scorecard-action` v2.4.4 and may therefore
continue to report no recognized fuzzing despite this bounded execution path.

After merging, inspect the scheduled Scorecards workflow and retain its actual
Fuzzing result. Closing the detection gap requires supported Python detection
upstream or a genuine OSS-Fuzz/ClusterFuzzLite integration. Do not add unused
configuration solely to influence the score.
