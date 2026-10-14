# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for safe pre-processing transform specifications."""

import pytest
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import InterpolationMode

import anomalib.models
from anomalib.data.transforms import ExportableCenterCrop, SquarePad
from anomalib.data.utils.generators.perlin import PerlinAnomalyGenerator
from anomalib.models import __all__ as model_names
from anomalib.pre_processing.utils._transform_registry import TRANSFORM_REGISTRY
from anomalib.pre_processing.utils.spec import spec_to_transform, transform_to_spec


@pytest.mark.parametrize(
    "transform",
    [
        v2.Compose([
            v2.Resize((128, 192), interpolation=v2.InterpolationMode.BICUBIC, antialias=False),
            v2.Normalize([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]),
        ]),
        v2.CenterCrop((96, 128)),
        v2.Grayscale(num_output_channels=3),
        ExportableCenterCrop((64, 80)),
        SquarePad(),
        v2.Compose([SquarePad(), v2.Resize((224, 224))]),
    ],
)
def test_transform_round_trip(transform: v2.Transform) -> None:
    """Transform specs reconstruct equivalent transforms."""
    spec = transform_to_spec(transform)
    reconstructed = spec_to_transform(spec)

    assert transform_to_spec(reconstructed) == spec


def test_none_round_trip() -> None:
    """None remains None."""
    assert transform_to_spec(None) is None
    assert spec_to_transform(None) is None


def test_rejects_unsafe_class_path() -> None:
    """Deserializer rejects imports outside approved namespaces."""
    with pytest.raises(ValueError, match="Unsupported transform class"):
        spec_to_transform({"class_path": "os.system", "init_args": {}})


def test_rejects_unsupported_transform() -> None:
    """Serializer rejects transforms outside approved namespaces."""

    class Unsupported(v2.Transform):
        """Test-only unsupported transform."""

        @staticmethod
        def transform(inpt: object, params: object) -> object:
            del params  # unused
            return inpt

    with pytest.raises(ValueError, match="Unsupported transform class"):
        transform_to_spec(Unsupported())


def test_rejects_unregistered_anomalib_transform() -> None:
    """Serializer rejects supported-namespace transforms outside the registry."""
    with pytest.raises(ValueError, match="Unsupported transform class"):
        transform_to_spec(PerlinAnomalyGenerator())


def test_first_party_transforms_canonicalize_to_public_export_path() -> None:
    """First-party transforms serialize to their package export, not their concrete module."""
    assert transform_to_spec(SquarePad())["class_path"] == "anomalib.data.transforms.SquarePad"
    assert transform_to_spec(ExportableCenterCrop(64))["class_path"] == "anomalib.data.transforms.ExportableCenterCrop"


def test_concrete_module_path_remains_a_readable_alias() -> None:
    """Specs written with the old concrete-module path still deserialize.

    Guards against a canonical-path change (e.g. moving a first-party
    transform's registration to its public export path) breaking checkpoints
    that were already saved under the previous canonical path.
    """
    legacy_spec = {
        "class_path": "anomalib.data.transforms.square_pad.SquarePad",
        "init_args": {},
    }

    reconstructed = spec_to_transform(legacy_spec)

    assert isinstance(reconstructed, SquarePad)


def test_registry_rejects_unregistered_class_path() -> None:
    """TRANSFORM_REGISTRY.class_for raises for an unregistered path."""
    with pytest.raises(ValueError, match="Unsupported transform class"):
        TRANSFORM_REGISTRY.class_for("os.system")


def test_registry_rejects_unregistered_class() -> None:
    """TRANSFORM_REGISTRY.path_for raises for an unregistered class."""

    class Unregistered(v2.Transform):
        """Test-only transform that is never registered."""

    with pytest.raises(ValueError, match="Unsupported transform class"):
        TRANSFORM_REGISTRY.path_for(Unregistered)


def test_enum_in_union_annotation_round_trips() -> None:
    """An enum constructor argument survives round-trip even inside a union annotation.

    ``Resize.interpolation`` is annotated as ``InterpolationMode | int | str``,
    not a bare ``InterpolationMode``. A restored transform must still carry a
    real ``InterpolationMode`` member (not the plain string written to the
    spec), matching the original transform's behaviour.
    """
    original = v2.Resize((64, 64), interpolation=InterpolationMode.BICUBIC)

    restored = spec_to_transform(transform_to_spec(original))

    assert isinstance(restored.interpolation, InterpolationMode)
    assert restored.interpolation == InterpolationMode.BICUBIC

    image = torch.rand(3, 32, 32)
    assert torch.equal(original(image), restored(image))


@pytest.mark.parametrize("model_name", model_names)
def test_every_model_default_preprocessor_round_trips(model_name: str) -> None:
    """Every registered model's default preprocessor can be saved and restored.

    Guards against models whose ``configure_pre_processor`` uses a transform
    outside the safe-serialization registry (e.g. ``SquarePad`` for L2BT),
    which would otherwise raise ``ValueError`` on every checkpoint save.

    Calls ``configure_pre_processor`` directly on the model *class* rather
    than instantiating the model (as ``AnomalibModule.__init__`` itself does
    via ``_resolve_component``, which calls this same classmethod/staticmethod
    with no arguments). Every model defines this as a classmethod or
    staticmethod, so this needs no model instance. Instantiating every model
    would run full constructors, which for some models (e.g. Draem's DTD
    download, EfficientAd's ImageNet download, timm backbones fetching
    pretrained weights) do network I/O and are unsuitable for a unit test.
    """
    model_cls = getattr(anomalib.models, model_name)
    processor = model_cls.configure_pre_processor()
    transform = None if processor is None else processor.transform

    spec = transform_to_spec(transform)

    assert transform_to_spec(spec_to_transform(spec)) == spec
