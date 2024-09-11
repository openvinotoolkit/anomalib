"""Tests for Custom Help Formatter."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import sys
from unittest.mock import patch

import pytest
from jsonargparse import ArgumentParser

from anomalib.cli.utils.help_formatter import (
    CustomHelpFormatter,
    get_cli_usage_docstring,
    get_verbose_usage,
    get_verbosity_subcommand,
    render_guide,
)


def test_get_verbosity_subcommand() -> None:
    """Test if the verbosity level and subcommand are correctly parsed."""
    argv = ["anomalib", "fit", "-h"]
    with patch.object(sys, "argv", argv):
        assert get_verbosity_subcommand() == {"help": True, "verbosity": 0, "subcommand": "fit"}

    argv = ["anomalib", "fit", "-h", "-v"]
    with patch.object(sys, "argv", argv):
        assert get_verbosity_subcommand() == {"help": True, "verbosity": 1, "subcommand": "fit"}

    argv = ["anomalib", "fit", "-h", "-vv"]
    with patch.object(sys, "argv", argv):
        assert get_verbosity_subcommand() == {"help": True, "verbosity": 2, "subcommand": "fit"}

    argv = ["anomalib", "-h"]
    with patch.object(sys, "argv", argv):
        assert get_verbosity_subcommand() == {"help": True, "verbosity": 2, "subcommand": None}


def test_get_verbose_usage() -> None:
    """Test if the verbose usage is correctly parsed."""
    subcommand = "test111"
    assert f"anomalib {subcommand} [optional_arguments]" in get_verbose_usage(subcommand=subcommand)


def test_get_cli_usage_docstring() -> None:
    """Test if the CLI usage docstring is correctly parsed."""
    assert get_cli_usage_docstring(None) is None

    class Component:
        """<Prev Section>.

        CLI Usage:
            1. First Step.
            2. Second Step.

        <Next Section>
        """

    assert get_cli_usage_docstring(Component) == "1. First Step.\n2. Second Step."

    class Component2:
        """<Prev Section>.

        CLI Usage-Test:
            test: test.

        <Next Section>
        """

    assert get_cli_usage_docstring(Component2) is None


def test_render_guide() -> None:
    """Test if the guide is correctly rendered."""
    subcommand = "fit"
    contents = render_guide(subcommand)
    assert len(contents) == 2
    assert contents[0].__class__.__name__ == "Markdown"
    assert "# Anomalib CLI Guide" in contents[0].markup
    assert contents[1].__class__.__name__ == "Panel"
    assert "anomalib fit --model anomalib.models.Padim" in contents[1].renderable.markup
    assert render_guide(None) == []


class TestCustomHelpFormatter:
    """Test Custom Help Formatter."""

    @pytest.fixture()
    @staticmethod
    def mock_parser() -> ArgumentParser:
        """Mock ArgumentParser."""
        parser = ArgumentParser(formatter_class=CustomHelpFormatter)
        parser.formatter_class.subcommand = "fit"
        parser.add_argument(
            "-t",
            "--test",
            action="count",
            help="add_usage test.",
        )
        parser.add_argument(
            "--model",
            action="count",
            help="never_skip test.",
        )
        return parser

    @staticmethod
    def test_verbose_0(capfd: "pytest.CaptureFixture", mock_parser: ArgumentParser) -> None:
        """Test verbose level 0."""
        argv = ["anomalib", "fit", "-h"]
        assert mock_parser.formatter_class == CustomHelpFormatter
        mock_parser.formatter_class.verbosity_level = 0
        with pytest.raises(SystemExit, match="0"):
            mock_parser.parse_args(argv)
        out, _ = capfd.readouterr()
        assert "Quick-Start" in out
        assert "Arguments" not in out

    @staticmethod
    def test_verbose_1(capfd: "pytest.CaptureFixture", mock_parser: ArgumentParser) -> None:
        """Test verbose level 1."""
        argv = ["anomalib", "fit", "-h", "-v"]
        assert mock_parser.formatter_class == CustomHelpFormatter
        mock_parser.formatter_class.verbosity_level = 1
        with pytest.raises(SystemExit, match="0"):
            mock_parser.parse_args(argv)
        out, _ = capfd.readouterr()
        assert "Quick-Start" in out
        assert "Arguments" in out

    @staticmethod
    def test_verbose_2(capfd: "pytest.CaptureFixture", mock_parser: ArgumentParser) -> None:
        """Test verbose level 2."""
        argv = ["anomalib", "fit", "-h", "-vv"]
        assert mock_parser.formatter_class == CustomHelpFormatter
        mock_parser.formatter_class.verbosity_level = 2
        with pytest.raises(SystemExit, match="0"):
            mock_parser.parse_args(argv)
        out, _ = capfd.readouterr()
        assert "Quick-Start" not in out
        assert "Arguments" in out
