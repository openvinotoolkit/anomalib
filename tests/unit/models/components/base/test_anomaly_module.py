"""Test AnomalibModule module."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from torch import nn

from anomalib.models.components.base import AnomalibModule


@pytest.fixture(scope="class")
def model_config_folder_path() -> str:
    """Fixture that returns model config folder path."""
    return "examples/configs/model"


class TestAnomalibModule:
    """Test AnomalibModule."""

    @pytest.fixture(autouse=True)
    def setup(self, model_config_folder_path: str) -> None:
        """Setup test AnomalibModule."""
        self.model_config_folder_path = model_config_folder_path

    @staticmethod
    def test_from_config_with_wrong_config_path() -> None:
        """Test AnomalibModule.from_config with wrong model name."""
        with pytest.raises(FileNotFoundError):
            AnomalibModule.from_config(config_path="wrong_configs.yaml")

    @pytest.mark.parametrize(
        "model_name",
        [
            "ai_vad",
            "cfa",
            "cflow",
            "csflow",
            "dfkde",
            "dfm",
            "draem",
            "dsr",
            "efficient_ad",
            "fastflow",
            "ganomaly",
            "padim",
            "patchcore",
            "reverse_distillation",
            "stfpm",
            "uflow",
        ],
    )
    def test_from_config(self, model_name: str) -> None:
        """Test AnomalibModule.from_config."""
        config_path = Path(self.model_config_folder_path) / f"{model_name}.yaml"
        model = AnomalibModule.from_config(config_path=config_path)
        assert model is not None
        assert isinstance(model, AnomalibModule)


class TestResolveComponents:
    """Test AnomalibModule._resolve_component."""

    class DummyComponent(nn.Module):
        """Dummy component class."""

        def __init__(self, value: int) -> None:
            self.value = value

    @classmethod
    def dummy_configure_component(cls) -> DummyComponent:
        """Dummy configure component method, simulates configure_<component> methods in module."""
        return cls.DummyComponent(value=1)

    def test_component_passed(self) -> None:
        """Test that the component is returned as is if it is an instance of the component type."""
        component = self.DummyComponent(value=0)
        resolved = AnomalibModule._resolve_component(  # noqa: SLF001
            component=component,
            component_type=self.DummyComponent,
            default_callable=self.dummy_configure_component,
        )
        assert isinstance(resolved, self.DummyComponent)
        assert resolved.value == 0

    def test_component_true(self) -> None:
        """Test that the default_callable is called if component is True."""
        component = True
        resolved = AnomalibModule._resolve_component(  # noqa: SLF001
            component=component,
            component_type=self.DummyComponent,
            default_callable=self.dummy_configure_component,
        )
        assert isinstance(resolved, self.DummyComponent)
        assert resolved.value == 1

    def test_component_false(self) -> None:
        """Test that None is returned if component is False."""
        component = False
        resolved = AnomalibModule._resolve_component(  # noqa: SLF001
            component=component,
            component_type=self.DummyComponent,
            default_callable=self.dummy_configure_component,
        )
        assert resolved is None

    def test_raises_type_error(self) -> None:
        """Test that a TypeError is raised if the component is not of the correct type."""
        component = 1
        with pytest.raises(TypeError):
            AnomalibModule._resolve_component(  # noqa: SLF001
                component=component,
                component_type=self.DummyComponent,
                default_callable=self.dummy_configure_component,
            )
