"""Test config upgrade entrypoint script."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from tools.upgrade.config import ConfigAdapter


class TestConfigAdapter:
    """This class contains test cases for the ConfigAdapter class.

    Methods:
        - test_config_adapter:
            Test case for upgrading and saving the original v0config to v1,
            and comparing it to the expected v1 config.
    """

    @staticmethod
    def test_config_adapter(project_path: Path) -> None:
        """Test the ConfigAdapter upgrade_all method.

        Test the ConfigAdapter class by upgrading and saving a v0 config to v1,
        and then comparing the upgraded config to the expected v1 config.

        Args:
            project_path (Path): The path to the project.

        Raises:
            AssertionError: If the upgraded config does not match the expected config.
        """
        original_config_path = Path(__file__).parent / "original_draem_v0.yaml"
        expected_config_path = Path(__file__).parent / "expected_draem_v1.yaml"
        upgraded_config_path = project_path / "upgraded_draem_v1.yaml"

        config_adapter = ConfigAdapter(original_config_path)

        # Upgrade and save the original v0 config to v1
        upgraded_config = config_adapter.upgrade_all()
        config_adapter.save_config(upgraded_config, upgraded_config_path)

        # Compare the upgraded config to the expected v1 config
        # Re-load the upgraded config from the saved file to ensure it is correctly saved
        upgraded_config = ConfigAdapter.safe_load(upgraded_config_path)
        expected_config = ConfigAdapter.safe_load(expected_config_path)
        assert upgraded_config == expected_config
