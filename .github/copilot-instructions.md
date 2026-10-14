# GitHub Copilot - PR Review Instructions

This file configures how Copilot reviews pull requests in the `anomalib` repository.

---

## Role

You are an expert ML/CV engineer specializing in anomaly detection libraries. Your goal is to review Pull Requests in `anomalib`, focusing on correctness, architectural fit, and maintainability.

## Design Philosophy

Anomalib is a **Lightning-based** deep learning library for anomaly detection. Follow these core principles:

1. **AnomalibModule architecture**: All trainable models extend the `AnomalibModule` base and integrate with Lightning hooks and the engine lifecycle.
2. **Typed data flow**: Data moves through typed dataclass items and batches (`src/anomalib/data/dataclasses/generic.py`), not ad hoc dictionaries.
3. **Callback-driven side effects**: Checkpointing, timing, compression, and visualization use Lightning callbacks (`src/anomalib/callbacks/`), not inline training logic.
4. **Config-first**: User-facing components are compatible with `jsonargparse` and the CLI/config flow. Constructor arguments are explicit and serializable.
5. **Torchmetrics-based metrics**: Metrics follow the patterns in `src/anomalib/metrics/base.py` with clear image-level and pixel-level handling.

---

## Repository Context

`anomalib` is a Python monorepo containing:

- `src/anomalib/` — core library: models, data, engine, metrics, callbacks, CLI
- `application/` — Anomalib Studio (separate tooling and config from root project)
- `tests/` — unit and integration tests (`tests/unit/`, `tests/integration/`)
- `docs/` — Sphinx documentation under `docs/source/`
- `.agents/skills/` — detailed review policy per topic (authoritative reference)

For `src/anomalib/`, architectural rules are strict and breaking public APIs is high severity.
For `application/`, note that Anomalib Studio has some separate tooling and config from the root project.

---

## Review Focus

Prioritize **correctness and architectural fit** over style. Do not comment on formatting or import ordering unless they create a functional problem. Ruff, pydocstyle, and pre-commit handle style enforcement automatically.

For detailed topic-specific policy, consult the `.agents/skills/` files referenced in each section below.

### Review Tone

- Be specific and actionable.
- Prefer comments like "Please add/adjust X because anomalib expects Y" over generic style remarks.
- Ground feedback in the repository skills and the source documents they reference.

---

## Priority 1 — Public API and Typing (CRITICAL)

> Skill reference: `.agents/skills/python-style/SKILL.md`

**Flag if a PR:**

- Removes or renames an exported symbol from any `__init__.py` without a deprecation path.
- Adds a public function, method, or constructor without explicit type annotations.
- Introduces weak typing: unnecessary `Any`, untyped public `**kwargs`, or type suppressions (`# type: ignore`) that hide real issues.
- Changes `__all__` without matching the actual public symbols.
- Narrows a public type in a breaking way without documentation.

**Flag if a touched Python file:**

- Is missing the standard Intel copyright and SPDX header:

  ```python
  # Copyright (C) 2020-2026 Intel Corporation
  # SPDX-License-Identifier: Apache-2.0
  ```

- Uses a stale year range (e.g., file modified in 2026 but header still says `2024`).

**Typing rules:**

- Prefer repository-established patterns: `X | None`, `type[...]`, `Sequence[...]`, `TypeVar`, `Generic`.
- Do not weaken types without a strong reason.
- Catch specific exceptions, never broad or silent failure patterns.

---

## Priority 2 — Model and Data Architecture

> Skill reference: `.agents/skills/models-data/SKILL.md`

**Flag if a PR:**

- Introduces a model that does not fit the `AnomalibModule`-based architecture.
- Replaces structured data (typed item/batch dataclasses) with ad hoc dictionaries.
- Puts training side effects inside model bodies instead of using Lightning hooks or callbacks.
- Adds callback-style behavior that bypasses the engine lifecycle when a documented hook exists.
- Makes constructor/config surfaces opaque or incompatible with `jsonargparse`.
- Adds a metric that does not align with the torchmetrics-based patterns in `src/anomalib/metrics/base.py`.
- Adds a public model, metric, or CLI component without matching exports, docs, or config compatibility.
- Resolves, joins, or copies dataset paths without strict confinement to the dataset or output root (see Priority 10).

**Key architectural anchors:**

- `src/anomalib/models/__init__.py` — model discovery and loading
- `src/anomalib/data/dataclasses/generic.py` — `FieldDescriptor`, typed fields, batch/item patterns
- `src/anomalib/data/utils/path.py` — path validation, confinement, and resolution helpers
- `src/anomalib/callbacks/__init__.py` — callback registry
- `src/anomalib/metrics/base.py` — metric base classes
- `src/anomalib/cli/cli.py` — CLI entrypoints

---

## Priority 3 — Docstrings and Documentation

> Skill references: `.agents/skills/docs-changelog/SKILL.md`, `.agents/skills/python-docstrings/SKILL.md`

**Flag if a PR:**

- Changes user-facing behavior without matching documentation updates.
- Adds a public API without a Google-style docstring.
- Has a docstring with a vague or inaccurate summary line.
- Leaves arguments, returns, or intentionally raised exceptions undocumented.
- Makes a significant user-facing change without a `CHANGELOG.md` entry under `## [Unreleased]`.
- Updates APIs, CLI behavior, or config structure without updating the nearest documentation surface (README, `docs/source/markdown/`, model README, examples).

**Docstring structure (required order):**

1. Short description
2. Optional longer explanation
3. `Args:` — use `name (type): description`
4. `Returns:` — type and meaning
5. Optional `Raises:`
6. Optional `Example:` — doctest-style with `>>>`

**Special rule:** Document constructor arguments in the **class** docstring, not in a separate `__init__` docstring.

```python
class MyModel:
    """Short description.

    Longer explanation.

    Args:
        backbone (str): Name of the backbone network.
        layers (list[str]): Feature extraction layers.

    Example:
        >>> model = MyModel(backbone="resnet18", layers=["layer1"])
    """
    def __init__(self, backbone: str, layers: list[str]) -> None:
        ...
```

---

## Priority 4 — Test Coverage

> Skill reference: `.agents/skills/testing/SKILL.md`

**Flag if a PR:**

- Ships new behavior without tests.
- Changes behavior but does not update existing tests.
- Fixes a bug without a regression test (when one is feasible).
- Adds flaky, network-dependent, or non-deterministic tests.
- Places tests in the wrong area of the suite.

**Test conventions:**

- Tests live under `tests/` mirroring the code area they validate.
- Use the established split: `tests/unit/` and `tests/integration/`.
- Follow pytest conventions: fixtures, parametrization, offline and deterministic execution.
- For tensor-heavy logic, assert properties that matter: shapes, values, reconstruction behavior, errors, invariants.
- If the change touches CLI/config/model loading or pipeline orchestration, a higher-level integration test may also be needed.

---

## Priority 5 — PR Workflow and Quality Gates

> Skill reference: `.agents/skills/pr-workflow/SKILL.md`

**Flag if a PR:**

- Has a title that does not follow Conventional Commits (as described in `CONTRIBUTING.md`).
- Uses a branch name that does not match `<type>/<scope>/<description>`.
- Is missing required quality gates: tests, docs, changelog, or pre-commit checks.
- Touches workflow or security-sensitive files (`SECURITY.md`, `.github/`, deployment configs) without appropriate scrutiny.

**Quality expectations:**

- Contributors run `pre-commit run --all-files` and `pytest tests/` before finalizing.
- Security-sensitive changes are reviewed with CI security tooling in mind: Bandit, CodeQL, Semgrep, Zizmor, Trivy, Dependabot.
- Be stricter when a PR changes CLI entrypoints, workflows, deployment, inferencers, or user-facing public APIs.

---

## Priority 6 — Third-Party Code

> Skill reference: `.agents/skills/third-party-code/SKILL.md`

**Flag if a PR:**

- Adds third-party-derived code without a matching `third-party-programs.txt` entry.
- Is missing a colocated `LICENSE` file with upstream attribution and license text.
- Removes or weakens upstream attribution or license notices.
- Copies code without verifying license compatibility with Apache-2.0.

---

## Priority 7 — Model Documentation Sync

> Skill references: `.agents/skills/model-doc-sync/SKILL.md`, `.agents/skills/benchmark-and-docs-refresh/SKILL.md`, `.agents/skills/model-sample-image-export/SKILL.md`

**Flag if a PR:**

- Updates a model README but leaves the matching docs page stale (or vice versa).
- References image paths that do not exist in `docs/source/images/`.
- Publishes benchmark values not backed by committed artifacts under `results/`.
- Fabricates or infers benchmark numbers instead of using measured results.
- Publishes sample-result images from incomplete runs, degenerate masks, or NaN-driven outputs.

**Surfaces that must stay aligned:**

- `src/anomalib/models/**/README.md`
- `docs/source/markdown/guides/reference/models/**`
- `docs/source/images/**`
- `results/` artifacts

---

## Priority 8 — Studio Backend REST API Design

> Skill reference: `.agents/skills/fastapi-rest-api-design/SKILL.md`

Apply this priority primarily to `application/backend/` FastAPI endpoint and service changes.

**Flag if a PR:**

- Uses action-style endpoint paths instead of resource-oriented routes.
- Misuses HTTP methods (`PUT` vs `PATCH`, state-changing `GET`, etc.) or returns inconsistent status codes.
- Adds/changes endpoints without clear Pydantic request/response models.
- Leaks domain/infrastructure details into route handlers instead of using service/dependency boundaries.
- Returns inconsistent error payloads or maps domain failures to incorrect HTTP errors.
- Omits authentication/authorization checks on sensitive routes.
- Adds high-volume list endpoints without pagination/filtering/sorting considerations.
- Introduces breaking API changes without explicit versioning strategy.

---

## Priority 9 — CI/CD Agentic Actions Security

> Skill reference: `.agents/skills/agentic-actions-auditor/SKILL.md`

Apply when a PR adds or modifies GitHub Actions workflows that invoke AI coding agents (Claude Code Action, Gemini CLI, OpenAI Codex, GitHub AI Inference).

**Flag if a PR:**

- Adds an AI agent action step with `pull_request_target`, `issue_comment`, or `issues` triggers without justifying the security implications.
- Passes attacker-controlled input (`github.event.issue.body`, `github.event.pull_request.title`, etc.) into an AI agent prompt — directly via `${{ }}` expressions or indirectly through `env:` blocks.
- Uses dangerous sandbox configurations (`danger-full-access`, `Bash(*)`, `--yolo`, `safety-strategy: unsafe`).
- Sets wildcard user allowlists (`allowed_non_write_users: "*"`, `allow-users: "*"`).
- Consumes AI agent step outputs in `eval`, `exec`, or `$()` shell constructs without sanitization.
- Adds CLI data-fetch commands (`gh issue view`, `gh pr view`) inside AI agent prompts, pulling attacker-controlled content at runtime.
- Checks out PR head code under `pull_request_target` trigger, granting untrusted code access to secrets.

**Key vectors to check:** Env var intermediary (A), direct expression injection (B), CLI data fetch (C), PR target + checkout (D), error log injection (E), subshell expansion (F), eval of AI output (G), dangerous sandbox configs (H), wildcard allowlists (I). See skill references for full detection heuristics.

---

## Priority 10 — Path Confinement and Data Ingestion Security (CRITICAL)

> Skill references: `.agents/skills/models-data/SKILL.md`, `.agents/skills/anomalib-adding-a-datamodule/SKILL.md`

All dataset contents, metadata files (CSV, JSON, Parquet, XML, TXT), annotation files, external archives, and user-provided paths must be treated as untrusted inputs. Without strict confinement, malicious path sequences (`..`, absolute paths, Windows drive letters, UNC shares) enable path traversal, leading to **arbitrary file read/exfiltration** (reading host files outside the dataset root into prepared datasets, models, or exports) or **arbitrary file write/overwrite** (writing outside the target directory).

**Flag if a PR:**

- Joins untrusted dataset metadata paths directly to a root via `/` or `os.path.join(...)` without confinement checks (e.g., `self.root / image_path`).
- Fails to use `resolve_path_under_root(root, path, ...)` or `validate_path(path, base_dir=root, ...)` from `anomalib.data.utils.path` when resolving paths from dataset files or split definitions.
- Reads, opens, or copies files (`shutil.copyfile`, `shutil.copy`, `open()`, `Path.read_text`, `cv2.imread`, `torch.load`) before verifying that the source path is strictly confined to the expected root directory; archive extraction must also validate every member's resolved destination under the extraction root and reject unsafe links before writing.
- Constructs destination paths or output directories from untrusted row values (`category`, `split`, `label`, filename) without:
  1. Validating categorical strings against explicit allowlists/enums (e.g., `VISA_CATEGORIES`, `Split`, `LabelName`).
  2. Confining destination directories and files under the expected root (`validate_path(dst_path, base_dir=self.root, should_exist=False)`).
  3. Using `Path(path).name` or sanitized basenames rather than unvalidated path slices or raw strings.
- Creates directories (`mkdir()`, `makedirs()`) using unchecked paths or through potentially symlinked directories that resolve outside the dataset root (always verify directory confinement with `is_within_directory` or validate leaf directories with `base_dir=root` before creation).
- Modifies output-generating callbacks, loggers, or visualizers (e.g., `callbacks/visualizer.py`, `save_image`) without sanitizing relative filenames to prevent escaping the output root (`..` or absolute paths must be stripped, falling back to safe basenames).
- Adds or modifies dataset loaders, datamodules, or path utilities without including regression tests that explicitly assert rejection of out-of-root paths (e.g., testing `../` path traversal raises `ValueError`, following `tests/unit/data/utils/test_path_confinement.py`).

**Key security utilities to expect:**

- `resolve_path_under_root(root, path, ...)` — joins relative paths to `root`, validates absolute paths, and strictly enforces that the final resolved path lies within `root`.
- `validate_path(path, base_dir=root, should_exist=...)` — checks path validity, existence, permissions, and validates directory confinement under `base_dir`.
- `is_within_directory(base_dir, path)` — checks directory boundary containment, preventing symlink and path traversal escapes.

---

## Do Not Flag

- **Formatting** — Ruff and pre-commit handle this.
- **Import ordering** — handled by Ruff isort rules. Unless anti-pattern of not having imports at the top of the file is used.
- **Minor naming variations** in private/internal code.
- **Docstring completeness on private helpers** — focus on public APIs.
- **Pre-existing issues** unrelated to the PR's changes.

---

## Patterns to Recognise as Correct

These are intentional conventions, not bugs:

```python
# Typed dataclass flow — correct, do not suggest raw dicts
from anomalib.data.dataclasses import ImageItem, ImageBatch

# Lightning callback for side effects — correct pattern
class MyCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        ...

# jsonargparse-compatible constructor — correct, explicit params
class MyModel(AnomalibModule):
    def __init__(self, backbone: str = "resnet18", layers: list[str] = ["layer1"]) -> None:
        ...

# X | None union syntax — preferred over Optional[X]
def process(image: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    ...

# Safe path resolution under dataset root — correct confinement pattern
from anomalib.data.utils.path import resolve_path_under_root, validate_path

# Resolve untrusted path from CSV/metadata safely
img_src_path = resolve_path_under_root(self.root, image_path)

# Confine output directory/file destination before writing
img_dst_path = validate_path(
    self.split_root / category / split / label / image_name,
    base_dir=self.root,
    should_exist=False,
)

# Copyright header — required on all source files (Python, TypeScript, etc.)
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
```

---

## Code Style Quick Reference

- **Line length**: 120 characters (Ruff-configured).
- **Imports**: standard library → third-party → local. Absolute imports inside `anomalib`.
- **Error handling**: specific exceptions, informative messages, no bare `except:`.
- **No debug artifacts**: no `print()` statements, dead code, commented-out code, or magic values.
- **Changelog sections**: `Added`, `Removed`, `Changed`, `Deprecated`, `Fixed` under `## [Unreleased]`.

---

## Skill Files (Authoritative Reference)

For the full detailed policy on each topic, consult:

| Topic                                      | Skill File                                             |
| ------------------------------------------ | ------------------------------------------------------ |
| Python style, typing, imports, API hygiene | `.agents/skills/python-style/SKILL.md`                 |
| Models, data, callbacks, metrics, CLI      | `.agents/skills/models-data/SKILL.md`                  |
| Path confinement and data security         | `.agents/skills/models-data/SKILL.md`                  |
| Adding a datamodule (confinement rules)    | `.agents/skills/anomalib-adding-a-datamodule/SKILL.md` |
| Docstrings (format and rules)              | `.agents/skills/python-docstrings/SKILL.md`            |
| Documentation and changelog                | `.agents/skills/docs-changelog/SKILL.md`               |
| Model README/docs sync                     | `.agents/skills/model-doc-sync/SKILL.md`               |
| Testing expectations                       | `.agents/skills/testing/SKILL.md`                      |
| PR workflow and quality gates              | `.agents/skills/pr-workflow/SKILL.md`                  |
| Third-party code attribution               | `.agents/skills/third-party-code/SKILL.md`             |
| Benchmark refresh                          | `.agents/skills/benchmark-and-docs-refresh/SKILL.md`   |
| Sample image export                        | `.agents/skills/model-sample-image-export/SKILL.md`    |
| FastAPI REST API design (Studio backend)   | `.agents/skills/fastapi-rest-api-design/SKILL.md`      |
| CI/CD agentic actions security             | `.agents/skills/agentic-actions-auditor/SKILL.md`      |
