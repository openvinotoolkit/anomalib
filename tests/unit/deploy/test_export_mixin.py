# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for ONNX export mixin behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import pytest
import torch
from lightning.pytorch import LightningModule

from anomalib.models.components.base.export_mixin import ExportMixin

if TYPE_CHECKING:
    from pathlib import Path


class DummyOutput(NamedTuple):
    """Dummy named tuple output for export tests."""

    pred_score: torch.Tensor
    anomaly_map: torch.Tensor | None


class DummyExportModel(ExportMixin, LightningModule):
    """Minimal exportable Lightning module for unit tests."""

    _weight: torch.nn.Parameter

    def __init__(self) -> None:
        super().__init__()
        self._weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: torch.Tensor) -> DummyOutput:
        """Return named tuple output expected by ``ExportMixin``.

        Args:
            batch (torch.Tensor): Input batch.

        Returns:
            DummyOutput: Dummy export output.
        """
        return DummyOutput(pred_score=batch.mean(dim=(1, 2, 3)) * self._weight, anomaly_map=None)


def test_to_onnx_uses_dynamo_exporter_by_default(mocker: pytest.MockFixture, tmp_path: Path) -> None:
    """Test that ONNX export uses ``dynamo=True`` by default."""
    export_mock = mocker.patch("torch.onnx.export")
    dim_type = type(torch.export.Dim("probe"))
    dim_mock = mocker.patch("torch.export.Dim", wraps=torch.export.Dim)
    model = DummyExportModel()

    model.to_onnx(tmp_path, input_size=(32, 32))

    assert export_mock.call_args.kwargs["dynamo"] is True
    dim_mock.assert_called_once_with("batch_size")
    assert isinstance(export_mock.call_args.kwargs["dynamic_shapes"][0][0], dim_type)
    assert "dynamic_axes" not in export_mock.call_args.kwargs


def test_to_onnx_treats_none_dynamo_as_true(mocker: pytest.MockFixture, tmp_path: Path) -> None:
    """Test that ``dynamo=None`` keeps the dynamo exporter enabled."""
    export_mock = mocker.patch("torch.onnx.export")
    model = DummyExportModel()

    model.to_onnx(tmp_path, input_size=(32, 32), dynamo=None)

    assert export_mock.call_args.kwargs["dynamo"] is True


def test_to_onnx_rejects_legacy_dynamo_false(mocker: pytest.MockFixture, tmp_path: Path) -> None:
    """``dynamo=False`` raises before creating the export folder or running the model."""
    model = DummyExportModel()
    forward = mocker.spy(model, "forward")

    with pytest.raises(ValueError, match=r"dynamo=False.*removed"):
        model.to_onnx(tmp_path, input_size=(32, 32), dynamo=False)

    forward.assert_not_called()
    assert not (tmp_path / "weights").exists()


def test_to_onnx_uses_dynamic_shapes_for_dynamo_export(mocker: pytest.MockFixture, tmp_path: Path) -> None:
    """Test that dynamo export receives ``dynamic_shapes`` instead of relying on conversion."""
    # ``Dim`` is a factory function on torch<2.7 and a class later; compare against what it returns.
    dim_type = type(torch.export.Dim("probe"))
    export_mock = mocker.patch("torch.onnx.export")
    dim_mock = mocker.patch("torch.export.Dim", wraps=torch.export.Dim)
    model = DummyExportModel()

    model.to_onnx(tmp_path, input_size=(32, 32), dynamo=True)

    assert export_mock.call_args.kwargs["dynamo"] is True
    dim_mock.assert_called_once_with("batch_size")
    assert isinstance(export_mock.call_args.kwargs["dynamic_shapes"][0][0], dim_type)
    assert "dynamic_axes" not in export_mock.call_args.kwargs


def test_to_onnx_translates_custom_dynamic_axes_for_dynamo_export(
    mocker: pytest.MockFixture,
    tmp_path: Path,
) -> None:
    """Test that custom ``dynamic_axes`` are converted to input-only ``dynamic_shapes`` for dynamo."""
    # ``Dim`` is a factory function on torch<2.7 and a class later; compare against what it returns.
    dim_type = type(torch.export.Dim("probe"))
    export_mock = mocker.patch("torch.onnx.export")
    dim_mock = mocker.patch("torch.export.Dim", wraps=torch.export.Dim)
    model = DummyExportModel()

    model.to_onnx(
        tmp_path,
        input_size=None,
        input_names=["image"],
        dynamo=True,
        dynamic_axes={"image": {0: "batch_size", 2: "height", 3: "width"}, "pred_score": {0: "batch_size"}},
    )

    shapes = export_mock.call_args.kwargs["dynamic_shapes"][0]
    assert set(shapes) == {0, 2, 3}
    assert all(isinstance(dim, dim_type) for dim in shapes.values())
    assert {call.args[0] for call in dim_mock.call_args_list} == {"batch_size", "height", "width"}
    assert "dynamic_axes" not in export_mock.call_args.kwargs


@pytest.mark.parametrize(
    ("input_size", "dynamic_shapes", "expected_shape"),
    [
        (None, {"batch": {0: torch.export.Dim("batch_size")}}, (2, 3, 32, 32)),
        (None, ([torch.export.Dim("batch_size"), None, None, None],), (2, 3, 32, 32)),
        (None, ({0: None},), (1, 3, 32, 32)),
        (None, ({0: 1},), (1, 3, 32, 32)),
        (None, ({0: torch.export.Dim.STATIC},), (1, 3, 32, 32)),
        ((40, 48), {}, (1, 3, 40, 48)),
    ],
)
def test_to_onnx_example_input_respects_dynamic_shape_forms(
    mocker: pytest.MockFixture,
    tmp_path: Path,
    input_size: tuple[int, int] | None,
    dynamic_shapes: object,
    expected_shape: tuple[int, ...],
) -> None:
    """Named, positional, sequence and static specs select compatible example dimensions."""
    export_mock = mocker.patch("torch.onnx.export")

    DummyExportModel().to_onnx(tmp_path, input_size=input_size, dynamic_shapes=dynamic_shapes)

    assert tuple(export_mock.call_args.kwargs["args"][0].shape) == expected_shape
    assert export_mock.call_args.kwargs["dynamic_shapes"] == dynamic_shapes


def test_to_onnx_raises_actionable_error_for_missing_onnxscript(
    mocker: pytest.MockFixture,
    tmp_path: Path,
) -> None:
    """Test that dynamo export failures mention ``onnxscript`` remediation."""
    model = DummyExportModel()
    export_mock = mocker.patch("torch.onnx.export")
    export_mock.side_effect = ModuleNotFoundError("No module named 'onnxscript'")

    with pytest.raises(ModuleNotFoundError, match="onnxscript") as exception:
        model.to_onnx(tmp_path, input_size=(32, 32), dynamo=True)

    assert "dynamo=False" not in str(exception.value)
    assert "anomalib[openvino]" in str(exception.value)


@pytest.mark.parametrize(
    ("export_kwargs", "expected_shape"),
    [
        ({"input_size": (40, 40)}, (2, 3, 40, 40)),
        ({}, (2, 3, 2, 2)),
        # Static exports keep batch 1, so the exported model accepts single images.
        ({"input_size": (40, 40), "dynamic_axes": None}, (1, 3, 40, 40)),
        ({"input_size": (40, 40), "dynamic_axes": {}}, (1, 3, 40, 40)),
        ({"input_size": (40, 40), "dynamic_shapes": ({},)}, (1, 3, 40, 40)),
        ({"dynamic_shapes": ({0: None},)}, (1, 3, 32, 32)),
        ({"dynamic_shapes": ({0: 1},)}, (1, 3, 32, 32)),
        ({"dynamic_shapes": ({0: torch.export.Dim.STATIC},)}, (1, 3, 32, 32)),
        (
            {
                "dynamic_shapes": {"batch": {0: torch.export.Dim("batch")}},
            },
            (2, 3, 32, 32),
        ),
        (
            {"dynamic_shapes": ([torch.export.Dim("batch"), None, None, None],)},
            (2, 3, 32, 32),
        ),
    ],
)
def test_to_onnx_example_input_matches_shape_spec(
    mocker: pytest.MockFixture,
    tmp_path: Path,
    export_kwargs: dict,
    expected_shape: tuple[int, ...],
) -> None:
    """Dynamic axes get example size 2 (dynamo fixes 0/1-sized examples); static axes keep their real size."""
    export_mock = mocker.patch("torch.onnx.export")

    DummyExportModel().to_onnx(tmp_path, **export_kwargs)

    assert tuple(export_mock.call_args.kwargs["args"][0].shape) == expected_shape


@pytest.mark.parametrize(("input_size", "run_shape"), [((32, 32), (3, 3, 32, 32)), (None, (3, 3, 64, 48))])
def test_to_onnx_dynamo_export_keeps_axes_dynamic(
    tmp_path: Path,
    input_size: tuple[int, int] | None,
    run_shape: tuple[int, ...],
) -> None:
    """A real dynamo export accepts batch sizes (and image sizes, without ``input_size``) other than the example."""
    pytest.importorskip("onnxscript")
    onnx = pytest.importorskip("onnx")
    ov = pytest.importorskip("openvino")
    model = DummyExportModel()

    onnx_path = model.to_onnx(tmp_path, input_size=input_size, dynamo=True)

    batch_dim = onnx.load(onnx_path).graph.input[0].type.tensor_type.shape.dim[0]
    assert batch_dim.dim_param, "batch axis was specialized to a fixed size"
    images = torch.rand(*run_shape)
    result = ov.Core().compile_model(str(onnx_path), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})(images.numpy())[0]
    torch.testing.assert_close(torch.from_numpy(result), model(images).pred_score)
