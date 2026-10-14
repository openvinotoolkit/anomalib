# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the Engine module."""

from pathlib import Path

from lightning import seed_everything

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim


class TestEngine:
    """Test Engine integration behavior."""

    @staticmethod
    def test_barebones_mode_metrics_and_checkpointing(tmp_path: Path, mvtecad_path: Path) -> None:
        """Test that barebones mode returns the same metrics and disables checkpointing.

        This test verifies that:
        1. Barebones mode and normal mode return the same metric values
        2. Both modes return the same set of metric keys
        3. Metrics are properly captured in barebones mode despite logging being disabled
        4. Normal mode (barebones=False) creates checkpoint files
        5. Barebones mode (barebones=True) does not create checkpoint files
        """
        datamodule = MVTecAD(root=mvtecad_path, category="dummy")

        # Test with normal mode
        seed_everything(42, workers=True)
        model_normal = Padim()
        engine_normal = Engine(default_root_dir=tmp_path / "normal")
        engine_normal.fit(model=model_normal, datamodule=datamodule)
        results_normal = engine_normal.test(model=model_normal, datamodule=datamodule)

        # Test with barebones mode
        seed_everything(42, workers=True)
        model_barebones = Padim()
        engine_barebones = Engine(default_root_dir=tmp_path / "barebones", barebones=True)
        engine_barebones.fit(model=model_barebones, datamodule=datamodule)
        results_barebones = engine_barebones.test(model=model_barebones, datamodule=datamodule)

        # Verify both modes return results
        assert results_normal
        assert results_barebones
        assert len(results_normal) > 0
        assert len(results_barebones) > 0

        # Extract metrics
        metrics_normal = results_normal[0]
        metrics_barebones = results_barebones[0]

        # Verify both have the same metric keys
        assert set(metrics_normal.keys()) == set(metrics_barebones.keys())

        # Verify expected metrics are present
        expected_metrics = {"image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score"}
        assert expected_metrics.issubset(set(metrics_normal.keys()))

        # Verify metric values are the same
        for metric_name in metrics_normal:
            value_normal = metrics_normal[metric_name]
            value_barebones = metrics_barebones[metric_name]

            if hasattr(value_normal, "item"):
                value_normal = value_normal.item()
            if hasattr(value_barebones, "item"):
                value_barebones = value_barebones.item()

            assert abs(value_normal - value_barebones) < 0.01

        # Verify checkpoint behavior
        normal_checkpoints = list((tmp_path / "normal").rglob("*.ckpt"))
        barebones_checkpoints = list((tmp_path / "barebones").rglob("*.ckpt"))

        # Verify normal mode (barebones=False) creates checkpoints
        assert len(normal_checkpoints) > 0, "Normal mode (barebones=False) should create checkpoint files"

        # Verify barebones mode (barebones=True) does not create checkpoints
        assert len(barebones_checkpoints) == 0, "Barebones mode (barebones=True) should not create checkpoint files"
