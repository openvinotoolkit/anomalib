# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for weights-only Engine checkpoint loading."""

from collections.abc import Callable
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.engine.plugins import AnomalibCheckpointIO
from anomalib.models import Padim


class TestCheckpointIO:
    """Test Engine checkpoint restore under the shared-fixture workflow."""

    @staticmethod
    def test_engine_test_loads_shared_checkpoint_weights_only(
        ckpt_path: Callable[[str], Path],
        project_path: Path,
        mvtecad_path: Path,
    ) -> None:
        """Engine ``ckpt_path`` restores via AnomalibCheckpointIO without a second fit.

        Uses the session-scoped ``ckpt_path`` resolver from ``tests.fixtures.models``
        so this does not retrain Padim when other integration modules already have.
        """
        checkpoint_path = ckpt_path("Padim")
        model = Padim()
        engine = Engine(
            default_root_dir=project_path,
            fast_dev_run=True,
            devices=1,
            logger=False,
        )
        datamodule = MVTecAD(root=mvtecad_path, category="dummy")

        results = engine.test(model=model, datamodule=datamodule, ckpt_path=str(checkpoint_path))

        assert results
        assert isinstance(engine.trainer.strategy.checkpoint_io, AnomalibCheckpointIO)

    @staticmethod
    def test_load_from_checkpoint_reads_shared_checkpoint(
        ckpt_path: Callable[[str], Path],
    ) -> None:
        """``load_from_checkpoint`` restores a trained Padim under weights_only=True."""
        checkpoint_path = ckpt_path("Padim")

        loaded = Padim.load_from_checkpoint(checkpoint_path)

        assert isinstance(loaded, Padim)
