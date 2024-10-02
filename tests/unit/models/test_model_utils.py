"""Test model utils."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from jsonargparse import Namespace
from omegaconf import OmegaConf

from anomalib.models import EfficientAd, Padim, Patchcore, UnknownModelError, get_model


class TestGetModel:
    """Test the `get_model` method."""

    @staticmethod
    def test_get_model_by_name() -> None:
        """Test get_model by name."""
        model = get_model("Padim")
        assert isinstance(model, Padim)
        model = get_model("padim")
        assert isinstance(model, Padim)

        model = get_model("EfficientAd")
        assert isinstance(model, EfficientAd)
        model = get_model("efficient_ad")
        assert isinstance(model, EfficientAd)
        model = get_model("efficientad")
        assert isinstance(model, EfficientAd)

    @staticmethod
    def test_get_model_by_name_with_init_args() -> None:
        """Test get_model by name with init args."""
        model = get_model("Patchcore", backbone="wide_resnet50_2")
        assert isinstance(model, Patchcore)

    @staticmethod
    def test_get_model_by_dict() -> None:
        """Test get_model by dict."""
        model = get_model({"class_path": "Padim"})
        assert isinstance(model, Padim)

    @staticmethod
    def test_get_model_by_dict_with_init_args() -> None:
        """Test get_model by dict with init args."""
        model = get_model({"class_path": "Padim", "init_args": {"backbone": "wide_resnet50_2"}})
        assert isinstance(model, Padim)
        model = get_model({"class_path": "Patchcore"}, backbone="wide_resnet50_2")
        assert isinstance(model, Patchcore)

    @staticmethod
    def test_get_model_by_dict_with_full_class_path() -> None:
        """Test get_model by dict with full class path."""
        model = get_model({"class_path": "anomalib.models.Padim", "init_args": {"backbone": "wide_resnet50_2"}})
        assert isinstance(model, Padim)

    @staticmethod
    def test_get_model_by_namespace() -> None:
        """Test get_model by namespace."""
        config = OmegaConf.create({"class_path": "Padim"})
        namespace = Namespace(**config)
        model = get_model(namespace)
        assert isinstance(model, Padim)

        # Argparse returns an object of type Namespace
        namespace = Namespace(
            class_path="anomalib.models.Padim",
            init_args=Namespace(
                layers=["layer1", "layer2", "layer3"],
                backbone="resnet18",
                pre_trained=True,
                n_features=None,
            ),
        )
        model = get_model(namespace)
        assert isinstance(model, Padim)

    @staticmethod
    def test_get_model_by_dict_config() -> None:
        """Test get_model by dict config."""
        config = OmegaConf.create({"class_path": "Padim"})
        model = get_model(config)
        assert isinstance(model, Padim)
        config = OmegaConf.create({"class_path": "Padim", "init_args": {"backbone": "wide_resnet50_2"}})
        model = get_model(config)
        assert isinstance(model, Padim)

    @staticmethod
    def test_get_unknown_model() -> None:
        """Test get_model with unknown model."""
        with pytest.raises(UnknownModelError):
            get_model("UnimplementedModel")

    @staticmethod
    def test_get_model_with_invalid_type() -> None:
        """Test get_model with invalid type."""
        with pytest.raises(TypeError):
            get_model(OmegaConf.create([{"class_path": "Padim"}]))

    @staticmethod
    def test_get_model_with_invalid_class_path() -> None:
        """Test get_model with invalid class path."""
        with pytest.raises(UnknownModelError):
            get_model({"class_path": "anomalib.models.InvalidModel"})
        with pytest.raises(UnknownModelError):
            get_model({"class_path": "InvalidModel"})
        with pytest.raises(UnknownModelError):
            get_model({"class_path": "anomalib.typo.InvalidModel"})
