# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test API.

Tests the models using API. The weight paths from the trained models are used for the rest of the tests.
"""

import contextlib
import sys
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import timm
from lightning import seed_everything
from lightning.pytorch.trainer.states import TrainerFn
from PIL import Image
from pytest_mock import MockerFixture

from anomalib.data import AnomalibDataModule, MVTec3D, MVTecAD
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import AnomalibModule, get_model, list_models

_FIT_CACHE: dict[str, Path] = {}
# ONNX files from ``test_export[onnx-*]``, reused by the OpenVINO export tests.
_ONNX_CACHE: dict[str, Path] = {}
_REAL_TIMM_CREATE_MODEL = timm.create_model


def _timm_create_model_offline(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Call timm.create_model with pretrained weights disabled."""
    kwargs["pretrained"] = False
    return _REAL_TIMM_CREATE_MODEL(*args, **kwargs)


def models() -> set[str]:
    """Return all available models."""
    return list_models()


def export_types() -> list[ExportType]:
    """Return all available export frameworks."""
    return list(ExportType)


@contextlib.contextmanager
def increased_recursion_limit(limit: int = 10000) -> Generator[None, None, None]:
    """Temporarily increase the recursion limit."""
    old_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(limit)
        yield
    finally:
        sys.setrecursionlimit(old_limit)


def _make_image_folder(root: Path, class_names: tuple[str, ...] = ("n01440764", "n02102040")) -> Path:
    """Create a tiny ``root/<class>/*.png`` tree to stand in for downloaded datasets.

    Used for EfficientAd's ImageNette (ImageFolder layout) and DRAEM's DTD textures
    (any image under the root), so CI never downloads multi-GB archives.
    """
    if root.is_dir():
        return root

    for class_name in class_names:
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(4):
            image = Image.fromarray(np.full((64, 64, 3), index * 40, dtype=np.uint8))
            image.save(class_dir / f"{index:03d}.png")
    return root


def _make_required_dataset(model_name: str, make_dummy_dataset: Callable[[str], Path]) -> None:
    """Generate the dummy dataset required by a model."""
    make_dummy_dataset("mvtecad")
    if model_name in {"cfm", "c_f_m"}:
        make_dummy_dataset("mvtec_3d")


def _ensure_fit(
    model_name: str,
    dataset_path: Path,
    project_path: Path,
) -> tuple[AnomalibModule, AnomalibDataModule, Engine, Path]:
    """Fit a model once and return fresh API objects with its cached checkpoint.

    The module cache is intentionally local to this test file. Stage tests must
    not delete cached checkpoints because later stages reuse them.
    """
    if model_name not in _FIT_CACHE:
        seed_everything(0, workers=True)
    model, dataset, engine = TestAPI._get_objects(model_name, dataset_path, project_path)  # noqa: SLF001
    if model_name not in _FIT_CACHE:
        engine.fit(model=model, datamodule=dataset)
        matches = list(
            project_path.glob(f"{model.name}/{dataset.name}/dummy/*/weights/lightning/model.ckpt"),
        )
        if not matches:
            msg = f"No checkpoint produced for {model_name}"
            raise FileNotFoundError(msg)
        _FIT_CACHE[model_name] = matches[0].resolve()
    return model, dataset, engine, _FIT_CACHE[model_name]


class TestAPI:
    """Do sanity check on all models."""

    @pytest.mark.parametrize("model_name", models())
    @staticmethod
    def test_fit(
        model_name: str,
        dataset_path: Path,
        project_path: Path,
        make_dummy_dataset: Callable[[str], Path],
    ) -> None:
        """Fit the model and save checkpoint.

        Args:
            model_name (str): Name of the model.
            dataset_path (Path): Root to dataset from fixture.
            project_path (Path): Path to temporary project folder from fixture.
            make_dummy_dataset (Callable[[str], Path]): Lazy dummy dataset factory.
        """
        _make_required_dataset(model_name, make_dummy_dataset)
        _ensure_fit(model_name, dataset_path, project_path)

    @pytest.mark.parametrize("model_name", models())
    @staticmethod
    def test_test(
        model_name: str,
        dataset_path: Path,
        project_path: Path,
        make_dummy_dataset: Callable[[str], Path],
    ) -> None:
        """Test model from checkpoint.

        Args:
            model_name (str): Name of the model.
            dataset_path (Path): Root to dataset from fixture.
            project_path (Path): Path to temporary project folder from fixture.
            make_dummy_dataset (Callable[[str], Path]): Lazy dummy dataset factory.
        """
        _make_required_dataset(model_name, make_dummy_dataset)
        model, dataset, engine, ckpt = _ensure_fit(model_name, dataset_path, project_path)
        engine.test(
            model=model,
            datamodule=dataset,
            ckpt_path=ckpt,
        )

    @pytest.mark.parametrize("model_name", models())
    @staticmethod
    def test_train(
        model_name: str,
        dataset_path: Path,
        project_path: Path,
        make_dummy_dataset: Callable[[str], Path],
    ) -> None:
        """Train model from checkpoint.

        Args:
            model_name (str): Name of the model.
            dataset_path (Path): Root to dataset from fixture.
            project_path (Path): Path to temporary project folder from fixture.
            make_dummy_dataset (Callable[[str], Path]): Lazy dummy dataset factory.
        """
        _make_required_dataset(model_name, make_dummy_dataset)
        model, dataset, _, ckpt = _ensure_fit(model_name, dataset_path, project_path)
        resume_engine = Engine(
            logger=False,
            default_root_dir=project_path,
            max_epochs=1,
            devices=1,
            limit_train_batches=1,
            limit_val_batches=2,
            limit_test_batches=2,
            max_steps=70000 if model_name == "efficient_ad" else -1,
        )
        resume_engine.train(
            model=model,
            datamodule=dataset,
            ckpt_path=ckpt,
        )

    @pytest.mark.parametrize("model_name", models())
    @staticmethod
    def test_validate(
        model_name: str,
        dataset_path: Path,
        project_path: Path,
        make_dummy_dataset: Callable[[str], Path],
    ) -> None:
        """Validate model from checkpoint.

        Args:
            model_name (str): Name of the model.
            dataset_path (Path): Root to dataset from fixture.
            project_path (Path): Path to temporary project folder from fixture.
            make_dummy_dataset (Callable[[str], Path]): Lazy dummy dataset factory.
        """
        _make_required_dataset(model_name, make_dummy_dataset)
        model, dataset, engine, ckpt = _ensure_fit(model_name, dataset_path, project_path)
        engine.validate(
            model=model,
            datamodule=dataset,
            ckpt_path=ckpt,
        )

    @pytest.mark.parametrize("model_name", models())
    @staticmethod
    def test_predict(
        model_name: str,
        dataset_path: Path,
        project_path: Path,
        make_dummy_dataset: Callable[[str], Path],
    ) -> None:
        """Predict using model from checkpoint.

        Args:
            model_name (str): Name of the model.
            dataset_path (Path): Root to dataset from fixture.
            project_path (Path): Path to temporary project folder from fixture.
            make_dummy_dataset (Callable[[str], Path]): Lazy dummy dataset factory.
        """
        _make_required_dataset(model_name, make_dummy_dataset)
        model, datamodule, engine, ckpt = _ensure_fit(model_name, dataset_path, project_path)
        engine.predict(
            model=model,
            ckpt_path=ckpt,
            datamodule=datamodule,
        )

    @pytest.mark.parametrize("model_name", models())
    @pytest.mark.parametrize("export_type", export_types())
    @staticmethod
    def test_export(
        model_name: str,
        export_type: ExportType,
        dataset_path: Path,
        project_path: Path,
        make_dummy_dataset: Callable[[str], Path],
        mocker: MockerFixture,
    ) -> None:
        """Export model from checkpoint.

        Args:
            model_name (str): Name of the model.
            export_type (ExportType): Framework to export to.
            dataset_path (Path): Root to dataset from fixture.
            project_path (Path): Path to temporary project folder from fixture.
            make_dummy_dataset (Callable[[str], Path]): Lazy dummy dataset factory.
            mocker (MockerFixture): Pytest mock fixture.
        """
        _make_required_dataset(model_name, make_dummy_dataset)
        model, _, engine, ckpt = _ensure_fit(model_name, dataset_path, project_path)

        # Some models require a fixed input size for ONNX export because they
        # use ops (e.g. kornia gaussian_blur2d) that ONNX cannot trace with
        # symbolic spatial dimensions.
        export_kwargs: dict[str, Any] = {}
        if model_name == "glass":
            export_kwargs["input_size"] = (288, 288)
        if model_name == "m_h_patchcore":
            export_kwargs["input_size"] = (224, 224)
        if model_name == "rad":
            # Positional bank is fitted on the 448x448 preprocessor crop.
            export_kwargs["input_size"] = (448, 448)
        if model_name == "found_a_d":
            export_kwargs["input_size"] = (224, 224)
        if model_name == "g_r_d_net":
            export_kwargs["input_size"] = (256, 256)
        if model_name in {"cfm", "c_f_m"}:
            export_kwargs["input_size"] = (224, 224)
            if export_type in {ExportType.ONNX, ExportType.OPENVINO}:
                pytest.skip("CFM uses dynamic point-cloud ops that are not ONNX/OpenVINO exportable")

        # Use context manager only for CSFlow
        with increased_recursion_limit() if model_name == "csflow" else contextlib.nullcontext():
            onnx_path = _ONNX_CACHE.get(model_name)
            if export_type == ExportType.OPENVINO and onnx_path is not None and onnx_path.exists():
                pytest.importorskip("openvino")
                # Reuse the ONNX graph while retaining coverage of Engine.export/to_openvino.
                mocker.patch.object(type(model), "to_onnx", return_value=onnx_path)

            exported_path = engine.export(
                model=model,
                ckpt_path=ckpt,
                export_type=export_type,
                model_file_name=model_name,
                **export_kwargs,
            )
            if export_type == ExportType.ONNX and exported_path is not None:
                _ONNX_CACHE[model_name] = exported_path

    @staticmethod
    def _get_objects(
        model_name: str,
        dataset_path: Path,
        project_path: Path,
    ) -> tuple[AnomalibModule, AnomalibDataModule, Engine]:
        """Return model, dataset, and engine objects.

        Args:
            model_name (str): Name of the model to train
            dataset_path (Path): Path to the root of dummy dataset
            project_path (Path): path to the temporary project folder

        Returns:
            tuple[AnomalibModule, AnomalibDataModule, Engine]: Returns the created objects for model, dataset,
                and engine
        """
        # set extra model args
        # TODO(ashwinvaidya17): Fix these Edge cases
        # https://github.com/open-edge-platform/anomalib/issues/1478

        extra_args = {}
        if model_name == "dfkde":
            extra_args["n_pca_components"] = 2
        if model_name == "m_h_patchcore":
            extra_args.update({
                "backbone": "resnet18",
                "pre_trained": False,
                "pca_variance_ratio": 0.1,
                "memory_bank_size": 16,
                "local_coreset_size": 8,
            })
        if model_name == "rad":
            # Avoid downloading the multi-GB DINOv3 checkpoint on CI.
            extra_args.update({
                "backbone": "vit_small_patch16_dinov3",
                "pre_trained": False,
                "layers": [3, 11],
                "k_image": 2,
            })
        if model_name == "found_a_d":
            # Keep the run small; pretrained download is blocked via patch below.
            # image_size must stay divisible by the encoder patch size (14).
            extra_args.update({
                "encoder_name": "dinov2_vit_small_14",
                "image_size": 224,
                "pred_depth": 2,
                "n_layer": 1,
                "top_k": 2,
                "use_few_shot_augmentation": False,
            })
        if model_name == "efficient_ad":
            # Avoid downloading the multi-GB ImageNette tarball on CI (~50+ min).
            extra_args["imagenet_dir"] = _make_image_folder(project_path / "efficient_ad_imagenette")
        if model_name == "draem":
            # Avoid downloading the DTD texture archive on CI.
            extra_args["dtd_dir"] = _make_image_folder(project_path / "dtd", class_names=("banded", "dotted"))
        if model_name in {"cfm", "c_f_m"}:
            # Keep integration tests lightweight/stable (point ops are memory hungry).
            extra_args["num_group"] = 128
            extra_args["group_size"] = 32

        if model_name in {"ai_vad", "fuvas"}:
            pytest.skip("Revisit video models tests")
        elif model_name in {"cfm", "c_f_m"}:
            dataset = MVTec3D(
                root=dataset_path / "mvtec_3d",
                category="dummy",
                # Keep parity with other models (and reduce runtime).
                train_batch_size=1,
            )
            dataset.setup(TrainerFn.FITTING)
            # Papers over NaN mask paths from the dummy generator/path validator until fixed upstream.
            dataset.train_data.samples["mask_path"] = dataset.train_data.samples["mask_path"].fillna("")
        else:
            # EfficientAd requires that the batch size be lesser than the number of images in the dataset.
            # This is so that the LR step size is not 0.
            dataset = MVTecAD(
                root=dataset_path / "mvtecad",
                category="dummy",
                # Keep resource-intensive models at batch size 1.
                train_batch_size=1 if model_name in {"efficient_ad", "g_r_d_net"} else 2,
                eval_batch_size=1 if model_name == "g_r_d_net" else 32,
            )

        # FoundAD hard-codes pretrained=True in its encoder loader; force offline
        # construction for CI without adding a public API knob.
        foundad_offline = (
            patch(
                "anomalib.models.image.foundad.components.encoder_loader.timm.create_model",
                side_effect=_timm_create_model_offline,
            )
            if model_name == "found_a_d"
            else contextlib.nullcontext()
        )
        with foundad_offline:
            model = get_model(model_name, **extra_args)

        if model_name == "vlm_ad":
            model.vlm_backend = MagicMock()
            model.vlm_backend.predict.return_value = "YES: Because reasons..."

        if model_name == "efficient_ad":
            max_steps = 70000
        elif model_name == "g_r_d_net":
            max_steps = 1
        else:
            max_steps = -1

        engine = Engine(
            logger=False,
            default_root_dir=project_path,
            max_epochs=1,
            devices=1,
            limit_train_batches=2,
            limit_val_batches=2,
            limit_test_batches=2,
            # TODO(ashwinvaidya17): Fix these Edge cases
            # https://github.com/open-edge-platform/anomalib/issues/1478
            max_steps=max_steps,
        )
        return model, dataset, engine
