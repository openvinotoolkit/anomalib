# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline nightly scheduling, artifact validation, and HF publication regression tests."""

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

SCRIPT = Path(__file__).resolve().parents[3] / ".github/scripts/benchmark/orchestrate.py"
SPEC = importlib.util.spec_from_file_location("nightly_benchmark", SCRIPT)
nightly = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = nightly
SPEC.loader.exec_module(nightly)
SHA = "a" * 40
STAMP = "2026-10-23-03_00_00"


def result(category: str = "bottle", timestamp: str = STAMP) -> dict:
    """One image-only pipeline result with deterministic provenance."""
    return {
        nightly.MODEL_COL: "Padim",
        nightly.DATA_COL: "MVTecAD",
        nightly.CATEGORY_COL: category,
        "anomalib_version": "2.7.0",
        "seed": 42,
        "image_AUROC": 0.9,
        "image_F1Score": 0.8,
        nightly.TIMESTAMP_COL: timestamp,
    }


@pytest.fixture
def categories(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid importing ML/data dependencies in scheduler-only tests."""
    monkeypatch.setattr(nightly, "default_categories", lambda _: ["bottle", "cable", "grid"])


@pytest.mark.parametrize(
    "override",
    [
        {"models": {"Padim": {}}},
        {"datasets": ["MVTecAD"]},
        {"num_categories": 0},
        {"num_categories": True},
        {"seed": "42"},
        {"trainer": []},
    ],
)
def test_matrix_validation(tmp_path: Path, override: dict) -> None:
    """YAML shape errors fail early rather than silently selecting a wrong workload."""
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump({"models": ["Padim"], "datasets": {"MVTecAD": {}}, **override}))
    with pytest.raises((TypeError, ValueError), match="matrix"):
        nightly.load_matrix(path)


@pytest.mark.parametrize(
    "models",
    [
        ["Padim", "Padim"],
        [{"name": "Padim", "init_args": []}],
        [{"name": "Padim", "data_init_args": {"category": "bottle"}}],
    ],
)
def test_invalid_models(models: list) -> None:
    """Reject duplicate models and overrides that bypass workload selection."""
    with pytest.raises((TypeError, ValueError)):
        nightly.parse_models(models)


@pytest.mark.usefixtures("categories")
@pytest.mark.parametrize("configured", [[], ["typo"], ["bottle", "bottle"], "bottle"])
def test_invalid_categories(configured: object) -> None:
    """Empty, misspelled, or duplicate subsets must not silently expand to all categories."""
    with pytest.raises(ValueError, match="Dataset"):
        nightly.expand_triples([nightly.ModelSpec("Padim")], {"MVTecAD": configured})


@pytest.mark.usefixtures("categories")
def test_missing_then_lru_selection() -> None:
    """Unmeasured triples have priority; completed triples rotate oldest first."""
    models = [nightly.ModelSpec("Padim")]
    triples = nightly.expand_triples(models, {"MVTecAD": {}})
    history = pd.DataFrame([result()])
    selection = nightly.select_workload(triples, models, history, 1)
    assert selection.categories == ["cable"]
    history = pd.DataFrame([result("bottle"), result("cable", "2026-10-21-03_00_00"), result("grid")])
    assert nightly.select_workload(triples, models, history, 2).categories == ["cable", "bottle"]


def test_legacy_history() -> None:
    """Legacy rows without run_timestamp are valid but oldest in scheduling."""
    history = pd.DataFrame([result()]).drop(columns=[nightly.TIMESTAMP_COL])
    assert nightly.history_timestamps(history)[nightly.Triple("Padim", "MVTecAD", "bottle")] == nightly.OLDEST_TS


def test_trainer_overrides(tmp_path: Path) -> None:
    """Per-model training budgets override global defaults without altering the selection."""
    spec = nightly.ModelSpec("EfficientAd", data_init_args={"train_batch_size": 1}, trainer={"max_steps": 1000})
    selection = nightly.Selection(spec, "MVTecAD", ["bottle"])
    path = nightly.write_benchmark_config(selection, {"trainer": {"max_epochs": 100}}, tmp_path, tmp_path / "job.yaml")
    config = yaml.safe_load(path.read_text())["benchmark"]
    assert config["trainer"] == {"max_epochs": 100, "max_steps": 1000}
    assert config["data"]["init_args"]["category"] == {"grid": ["bottle"]}
    assert config["model"]["init_args"]["imagenet_dir"] == str(tmp_path / "imagenette")


def test_pin_download_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Read data at a resolved commit SHA, not a moving branch or cached credential."""
    path = tmp_path / "remote.csv"
    pd.DataFrame([result()]).to_csv(path, index=False)
    api = MagicMock()
    api.dataset_info.return_value.sha = SHA
    constructor = MagicMock(return_value=api)
    download = MagicMock(return_value=str(path))
    monkeypatch.setattr(nightly, "HfApi", constructor)
    monkeypatch.setattr(nightly, "hf_hub_download", download)
    frame, revision = nightly.download_hf_results("org/data", tmp_path / "history.csv")
    assert len(frame) == 1
    assert revision == SHA
    assert download.call_args.kwargs["revision"] == SHA
    assert download.call_args.kwargs["token"] is False


@pytest.mark.parametrize(
    ("error", "empty"),
    [
        (EntryNotFoundError("no file"), True),
        (RepositoryNotFoundError("bad repo", response=MagicMock()), False),
        (OSError("offline"), False),
    ],
)
def test_download_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, empty: bool) -> None:
    """Only missing files mean new history; network/auth/repository failures stop publication."""
    api = MagicMock()
    api.dataset_info.return_value.sha = SHA
    monkeypatch.setattr(nightly, "HfApi", MagicMock(return_value=api))
    monkeypatch.setattr(nightly, "hf_hub_download", MagicMock(side_effect=error))
    if empty:
        frame, _ = nightly.download_hf_results("org/data", tmp_path / "history.csv")
        assert frame.empty
    else:
        with pytest.raises(type(error)):
            nightly.download_hf_results("org/data", tmp_path / "history.csv")


def test_complete_output_required(tmp_path: Path) -> None:
    """Pipeline partial failures cannot be published even when the process exits zero."""
    csv_path = tmp_path / "results.csv"
    pd.DataFrame([result()]).to_csv(csv_path, index=False)
    selection = nightly.Selection(nightly.ModelSpec("Padim"), "MVTecAD", ["bottle", "grid"])
    with pytest.raises(ValueError, match="selected workload"):
        nightly.prepare_new_rows(csv_path, STAMP, selection)
    selection.categories = ["bottle"]
    assert len(nightly.prepare_new_rows(csv_path, STAMP, selection)) == 1


def test_no_stale_output(tmp_path: Path) -> None:
    """Results outside the current execution directory cannot satisfy output discovery."""
    stale = tmp_path / "old" / STAMP
    stale.mkdir(parents=True)
    pd.DataFrame([result()]).to_csv(stale / "results.csv", index=False)
    with pytest.raises(FileNotFoundError):
        nightly.find_latest_results_csv(tmp_path / "fresh" / "runs" / "benchmark")


def test_invalid_scores_are_not_published() -> None:
    """Invalid numeric metrics cannot contaminate the public history."""
    frame = pd.DataFrame([result()])
    frame.loc[0, "image_AUROC"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        nightly.validate_results(frame)


def test_training_uses_current_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Training uses the already-locked interpreter and receives no write credentials."""
    run = MagicMock()
    monkeypatch.setenv("HF_TOKEN", "test-only-not-a-real-token")
    monkeypatch.setattr(nightly.subprocess, "run", run)
    nightly.run_benchmark(tmp_path / "config.yaml", tmp_path)
    args, kwargs = run.call_args
    assert args[0][0] == sys.executable
    assert "uv" not in args[0]
    assert kwargs["env"]["HF_TOKEN"] == ""
    assert kwargs["cwd"] == tmp_path


def test_upload_is_idempotent_and_guarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrying published identities is a no-op; new appends guard the snapshot revision."""
    frame = pd.DataFrame([result()])
    monkeypatch.setattr(nightly, "download_hf_results", MagicMock(return_value=(frame, SHA)))
    api = MagicMock()
    monkeypatch.setattr(nightly, "HfApi", MagicMock(return_value=api))
    nightly.append_and_upload(frame, "org/data", "test-token", tmp_path / "merged.csv")
    api.upload_file.assert_not_called()
    new = pd.DataFrame([result(timestamp="2026-10-24-03_00_00")])
    nightly.append_and_upload(new, "org/data", "test-token", tmp_path / "merged.csv")
    assert api.upload_file.call_args.kwargs["parent_commit"] == SHA
    assert len(pd.read_csv(tmp_path / "merged.csv")) == 2
    api.upload_file.side_effect = RuntimeError("concurrent update")
    with pytest.raises(RuntimeError, match="concurrent"):
        nightly.append_and_upload(new, "org/data", "test-token", tmp_path / "merged.csv")


def test_upload_requires_secret_before_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An absent write secret fails before trying a history download or upload."""
    download = MagicMock()
    monkeypatch.setattr(nightly, "download_hf_results", download)
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        nightly.append_and_upload(pd.DataFrame([result()]), "org/data", None, tmp_path / "merged.csv")
    download.assert_not_called()


@pytest.mark.usefixtures("categories")
def test_main_never_uploads_during_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run/selection mode stages complete output for a separate authenticated command."""
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump({"models": ["Padim"], "datasets": {"MVTecAD": ["bottle"]}}))
    monkeypatch.setattr(
        nightly,
        "parse_args",
        lambda: Namespace(
            matrix=matrix,
            work_dir=tmp_path,
            upload_results=None,
            hf_repo="org/data",
            num_categories=1,
            datasets_root=tmp_path / "datasets",
            dry_run=False,
            skip_upload=False,
        ),
    )
    monkeypatch.setattr(nightly, "download_hf_results", MagicMock(return_value=(pd.DataFrame(), SHA)))

    def benchmark(_config: Path, run_dir: Path) -> None:
        output = run_dir / "runs" / "benchmark" / STAMP / "results.csv"
        output.parent.mkdir(parents=True)
        pd.DataFrame([result()]).to_csv(output, index=False)

    monkeypatch.setattr(nightly, "run_benchmark", benchmark)
    upload = MagicMock()
    monkeypatch.setattr(nightly, "append_and_upload", upload)
    assert nightly.main() == 0
    assert (tmp_path / "new_results.csv").is_file()
    upload.assert_not_called()
