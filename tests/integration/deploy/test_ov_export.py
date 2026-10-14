# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenVINO export with different compression types."""

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import MVTecAD
from anomalib.deploy import CompressionType, ExportType
from anomalib.engine import Engine
from anomalib.metrics import AUROC
from anomalib.models import Padim


class TestOpenVINOExport:
    """Test OpenVINO export functionality with different compression options."""

    @pytest.fixture()
    @staticmethod
    def setup_model_and_data(
        mvtecad_path: Path,
        project_path: Path,
        ckpt_path: Callable[[str], Path],
    ) -> tuple[Padim, MVTecAD, Engine, Path, Path]:
        """Set up model, datamodule, engine, and checkpoint for testing.

        Args:
            mvtecad_path: Path to MVTec AD dataset from fixture
            project_path: Path to temporary project folder from fixture
            ckpt_path: Callable resolving a shared trained model checkpoint

        Returns:
            Tuple of (model, datamodule, engine, export_root, checkpoint_path)
        """
        model = Padim()
        datamodule = MVTecAD(root=mvtecad_path, category="dummy", augmentations=Resize((256, 256)))
        engine = Engine(default_root_dir=project_path)
        export_root = project_path / "exports"
        return model, datamodule, engine, export_root, ckpt_path("Padim")

    @staticmethod
    def test_export_openvino_no_compression(setup_model_and_data: tuple) -> None:
        """Test OpenVINO export without compression.

        Args:
            setup_model_and_data: Fixture providing model, datamodule, engine, and export_root
        """
        model, _, engine, export_root, checkpoint_path = setup_model_and_data

        exported_path = engine.export(
            model=model,
            export_type=ExportType.OPENVINO,
            export_root=export_root,
            ckpt_path=checkpoint_path,
            model_file_name="model_no_compression",
            input_size=(256, 256),
        )

        assert exported_path is not None
        assert exported_path.exists()
        assert exported_path.suffix == ".xml"
        assert exported_path.name == "model_no_compression.xml"
        assert (exported_path.parent / exported_path.stem).with_suffix(".bin").exists()

    @staticmethod
    def test_export_openvino_fp16(setup_model_and_data: tuple) -> None:
        """Test OpenVINO export with FP16 compression.

        Args:
            setup_model_and_data: Fixture providing model, datamodule, engine, and export_root
        """
        model, _, engine, export_root, checkpoint_path = setup_model_and_data

        exported_path = engine.export(
            model=model,
            export_type=ExportType.OPENVINO,
            export_root=export_root,
            ckpt_path=checkpoint_path,
            model_file_name="model_fp16",
            input_size=(256, 256),
            compression_type=CompressionType.FP16,
        )

        assert exported_path is not None
        assert exported_path.exists()
        assert exported_path.suffix == ".xml"
        assert exported_path.name == "model_fp16.xml"
        bin_file = (exported_path.parent / exported_path.stem).with_suffix(".bin")
        assert bin_file.exists()

    @pytest.mark.skipif(
        not pytest.importorskip("nncf", reason="NNCF not installed"),
        reason="NNCF required for INT8 compression",
    )
    @staticmethod
    def test_export_openvino_int8(setup_model_and_data: tuple) -> None:
        """Test OpenVINO export with INT8 weight compression.

        Args:
            setup_model_and_data: Fixture providing model, datamodule, engine, and export_root
        """
        model, _, engine, export_root, checkpoint_path = setup_model_and_data

        exported_path = engine.export(
            model=model,
            export_type=ExportType.OPENVINO,
            export_root=export_root,
            ckpt_path=checkpoint_path,
            model_file_name="model_int8",
            input_size=(256, 256),
            compression_type=CompressionType.INT8,
        )

        assert exported_path is not None
        assert exported_path.exists()
        assert exported_path.suffix == ".xml"
        assert exported_path.name == "model_int8.xml"

    @pytest.mark.skipif(
        not pytest.importorskip("nncf", reason="NNCF not installed"),
        reason="NNCF required for INT8_PTQ compression",
    )
    @staticmethod
    def test_export_openvino_int8_ptq(setup_model_and_data: tuple) -> None:
        """Test OpenVINO export with INT8 post-training quantization.

        Args:
            setup_model_and_data: Fixture providing model, datamodule, engine, and export_root
        """
        model, datamodule, engine, export_root, checkpoint_path = setup_model_and_data

        exported_path = engine.export(
            model=model,
            export_type=ExportType.OPENVINO,
            export_root=export_root,
            ckpt_path=checkpoint_path,
            model_file_name="model_int8_ptq",
            input_size=(256, 256),
            compression_type=CompressionType.INT8_PTQ,
            datamodule=datamodule,
        )

        assert exported_path is not None
        assert exported_path.exists()
        assert exported_path.suffix == ".xml"
        assert exported_path.name == "model_int8_ptq.xml"

    @pytest.mark.skipif(
        not pytest.importorskip("nncf", reason="NNCF not installed"),
        reason="NNCF required for INT8_ACQ compression",
    )
    @pytest.mark.skipif(
        os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true",
        reason="Skipping in CI due to high RAM requirements (>16GB). Test can cause segmentation faults.",
    )
    @staticmethod
    def test_export_openvino_int8_acq_default_metric(setup_model_and_data: tuple) -> None:
        """Test OpenVINO export with INT8 accuracy-control quantization using default metric.

        This test is marked as slow and may require significant RAM (>16GB).
        It is automatically skipped in CI environments to prevent segmentation faults.

        Args:
            setup_model_and_data: Fixture providing model, datamodule, engine, and export_root
        """
        model, datamodule, engine, export_root, checkpoint_path = setup_model_and_data

        # Test with default metric (should use F1Score automatically)
        exported_path = engine.export(
            model=model,
            export_type=ExportType.OPENVINO,
            export_root=export_root,
            ckpt_path=checkpoint_path,
            model_file_name="model_int8_acq_default",
            input_size=(256, 256),
            compression_type=CompressionType.INT8_ACQ,
            datamodule=datamodule,
            # metric=None (default) - should use F1Score
        )

        assert exported_path is not None
        assert exported_path.exists()
        assert exported_path.suffix == ".xml"
        assert exported_path.name == "model_int8_acq_default.xml"

    @pytest.mark.skipif(
        not pytest.importorskip("nncf", reason="NNCF not installed"),
        reason="NNCF required for INT8_ACQ compression",
    )
    @pytest.mark.skipif(
        os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true",
        reason="Skipping in CI due to high RAM requirements (>16GB). Test can cause segmentation faults.",
    )
    @staticmethod
    def test_export_openvino_int8_acq_custom_metric(setup_model_and_data: tuple) -> None:
        """Test OpenVINO export with INT8 accuracy-control quantization using custom metric.

        This test is marked as slow and may require significant RAM (>16GB).
        It is automatically skipped in CI environments to prevent segmentation faults.

        Args:
            setup_model_and_data: Fixture providing model, datamodule, engine, and export_root
        """
        model, datamodule, engine, export_root, checkpoint_path = setup_model_and_data

        # Test with custom metric
        metric = AUROC(fields=["pred_score", "gt_label"])
        exported_path = engine.export(
            model=model,
            export_type=ExportType.OPENVINO,
            export_root=export_root,
            ckpt_path=checkpoint_path,
            model_file_name="model_int8_acq_custom_metric",
            input_size=(256, 256),
            compression_type=CompressionType.INT8_ACQ,
            datamodule=datamodule,
            metric=metric,
        )

        assert exported_path is not None
        assert exported_path.exists()
        assert exported_path.suffix == ".xml"
        assert exported_path.name == "model_int8_acq_custom_metric.xml"

    @pytest.mark.skipif(
        not pytest.importorskip("nncf", reason="NNCF not installed"),
        reason="NNCF required for INT8_ACQ compression",
    )
    @pytest.mark.skipif(
        os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true",
        reason="Skipping in CI due to high RAM requirements (>16GB). Test can cause segmentation faults.",
    )
    @staticmethod
    def test_export_openvino_int8_acq_custom_max_drop(setup_model_and_data: tuple) -> None:
        """Test OpenVINO export with INT8 accuracy-control quantization using custom max_drop.

        This test is marked as slow and may require significant RAM (>16GB).
        It is automatically skipped in CI environments to prevent segmentation faults.

        Args:
            setup_model_and_data: Fixture providing model, datamodule, engine, and export_root
        """
        model, datamodule, engine, export_root, checkpoint_path = setup_model_and_data

        # Test with custom max_drop
        exported_path = engine.export(
            model=model,
            export_type=ExportType.OPENVINO,
            export_root=export_root,
            ckpt_path=checkpoint_path,
            model_file_name="model_int8_acq_max_drop",
            input_size=(256, 256),
            compression_type=CompressionType.INT8_ACQ,
            datamodule=datamodule,
            max_drop=0.02,  # Allow 2% accuracy drop
        )

        assert exported_path is not None
        assert exported_path.exists()
        assert exported_path.suffix == ".xml"
        assert exported_path.name == "model_int8_acq_max_drop.xml"

    @staticmethod
    def test_export_openvino_int8_ptq_missing_datamodule(setup_model_and_data: tuple) -> None:
        """Test that INT8_PTQ export raises error without datamodule.

        Args:
            setup_model_and_data: Fixture providing model, datamodule, engine, and export_root
        """
        model, _, engine, export_root, checkpoint_path = setup_model_and_data

        with pytest.raises(ValueError, match="Datamodule must be provided"):
            engine.export(
                model=model,
                export_type=ExportType.OPENVINO,
                export_root=export_root,
                ckpt_path=checkpoint_path,
                input_size=(256, 256),
                compression_type=CompressionType.INT8_PTQ,
                # datamodule not provided
            )

    @staticmethod
    def test_export_openvino_int8_acq_missing_datamodule(setup_model_and_data: tuple) -> None:
        """Test that INT8_ACQ export raises error without datamodule.

        Args:
            setup_model_and_data: Fixture providing model, datamodule, engine, and export_root
        """
        model, _, engine, export_root, checkpoint_path = setup_model_and_data

        with pytest.raises(ValueError, match="Datamodule must be provided"):
            engine.export(
                model=model,
                export_type=ExportType.OPENVINO,
                export_root=export_root,
                ckpt_path=checkpoint_path,
                input_size=(256, 256),
                compression_type=CompressionType.INT8_ACQ,
                # datamodule not provided
            )

    @pytest.mark.skipif(
        not pytest.importorskip("nncf", reason="NNCF not installed"),
        reason="NNCF required for INT8_ACQ compression",
    )
    @staticmethod
    def test_export_openvino_int8_acq_invalid_max_drop(setup_model_and_data: tuple) -> None:
        """Test that INT8_ACQ export raises error with invalid max_drop values.

        Args:
            setup_model_and_data: Fixture providing model, datamodule, engine, and export_root
        """
        model, datamodule, engine, export_root, checkpoint_path = setup_model_and_data

        # Test max_drop > 1
        with pytest.raises(ValueError, match="max_drop must be between 0 and 1"):
            engine.export(
                model=model,
                export_type=ExportType.OPENVINO,
                export_root=export_root,
                ckpt_path=checkpoint_path,
                input_size=(256, 256),
                compression_type=CompressionType.INT8_ACQ,
                datamodule=datamodule,
                max_drop=1.5,
            )

        # Test max_drop < 0
        with pytest.raises(ValueError, match="max_drop must be between 0 and 1"):
            engine.export(
                model=model,
                export_type=ExportType.OPENVINO,
                export_root=export_root,
                ckpt_path=checkpoint_path,
                input_size=(256, 256),
                compression_type=CompressionType.INT8_ACQ,
                datamodule=datamodule,
                max_drop=-0.1,
            )

    @staticmethod
    @staticmethod
    def test_export_openvino_max_drop_warning_wrong_compression(
        setup_model_and_data: tuple,
    ) -> None:
        """Test that max_drop parameter warns when used with non-INT8_ACQ compression.

        Args:
            setup_model_and_data: Fixture providing model, datamodule, engine, and export_root
        """
        model, _, engine, export_root, checkpoint_path = setup_model_and_data

        # Should warn when max_drop is provided but compression type is not INT8_ACQ
        with pytest.warns(UserWarning, match="max_drop parameter is only used for CompressionType.INT8_ACQ"):
            engine.export(
                model=model,
                export_type=ExportType.OPENVINO,
                export_root=export_root,
                ckpt_path=checkpoint_path,
                input_size=(256, 256),
                compression_type=CompressionType.FP16,
                max_drop=0.05,  # This should trigger warning
            )

    @pytest.mark.skipif(
        not pytest.importorskip("nncf", reason="NNCF not installed"),
        reason="NNCF required for INT8_ACQ compression",
    )
    @pytest.mark.skipif(
        os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true",
        reason="Skipping in CI due to high RAM requirements (>16GB). Test can cause segmentation faults.",
    )
    @staticmethod
    @staticmethod
    def test_export_openvino_max_drop_large_value_warning(
        setup_model_and_data: tuple,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that large max_drop values trigger a warning.

        This test is marked as slow and may require significant RAM (>16GB).
        It is automatically skipped in CI environments to prevent segmentation faults.

        Args:
            setup_model_and_data: Fixture providing model, datamodule, engine, and export_root
            caplog: Pytest fixture to capture log messages
        """
        model, datamodule, engine, export_root, checkpoint_path = setup_model_and_data

        # Should warn when max_drop > 0.1
        exported_path = engine.export(
            model=model,
            export_type=ExportType.OPENVINO,
            export_root=export_root,
            ckpt_path=checkpoint_path,
            model_file_name="model_int8_acq_large_drop",
            input_size=(256, 256),
            compression_type=CompressionType.INT8_ACQ,
            datamodule=datamodule,
            max_drop=0.15,  # Large value should trigger warning
        )

        # Check that warning was logged
        assert any("is a large value" in record.message for record in caplog.records)
        # Verify export succeeded despite warning
        assert exported_path is not None
        assert exported_path.exists()
        assert exported_path.name == "model_int8_acq_large_drop.xml"

    @staticmethod
    def test_export_openvino_with_ov_kwargs(setup_model_and_data: tuple) -> None:
        """Test OpenVINO export with custom OpenVINO kwargs.

        Args:
            setup_model_and_data: Fixture providing model, datamodule, engine, and export_root
        """
        model, _, engine, export_root, checkpoint_path = setup_model_and_data

        exported_path = engine.export(
            model=model,
            export_type=ExportType.OPENVINO,
            export_root=export_root,
            ckpt_path=checkpoint_path,
            model_file_name="model_with_kwargs",
            input_size=(256, 256),
            ov_kwargs={},  # Custom OpenVINO options
        )

        assert exported_path is not None
        assert exported_path.exists()
        assert exported_path.name == "model_with_kwargs.xml"
