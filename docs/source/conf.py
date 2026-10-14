"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Project information -----------------------------------------------------
https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""

# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
import sys
from pathlib import Path
from typing import Any

from docutils.parsers.rst import Directive, directives  # type: ignore[import-untyped]

# Define paths
project_root = Path(__file__).parent.parent.parent
module_path = project_root / "src"
examples_path = project_root / "examples"

# Insert paths to sys.path
sys.path.insert(0, str(module_path.resolve()))
sys.path.insert(0, str(project_root.resolve()))

project = "Anomalib"
copyright = "Intel Corporation"  # noqa: A001
author = "Intel Corporation"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
]

# Enable nbsphinx only if pandoc is available in the environment
if shutil.which("pandoc"):
    extensions.append("nbsphinx")

# MyST configuration
myst_enable_extensions = [
    "colon_fence",
    "linkify",
    "substitution",
    "tasklist",
    "deflist",
    "fieldlist",
    "amsmath",
    "dollarmath",
]

# Add separate setting for eval-rst
myst_enable_eval_rst = True

# Notebook handling
nbsphinx_allow_errors = True
nbsphinx_execute = "never"  # Pre-rendered notebook build in RTD without dynamic re-execution
nbsphinx_timeout = 300  # Timeout in seconds

# Templates and patterns
templates_path = ["_templates"]
exclude_patterns: list[str] = [
    "_build",
    "**.ipynb_checkpoints",
    "**/.pytest_cache",
    "**/.git",
    "**/.github",
    "**/.venv",
    "**/*.egg-info",
    "**/build",
    "**/dist",
    "examples/configs/README.md",
    "examples/notebooks/**README.md",
    "examples/notebooks/**.ipynb",
    "examples/notebooks/*/**.ipynb",
]

# Automatic exclusion of prompts from the copies
# https://sphinx-copybutton.readthedocs.io/en/latest/use.html#automatic-exclusion-of-prompts-from-the-copies
copybutton_exclude = ".linenos, .gp, .go"

# Enable section anchors for cross-referencing
autosectionlabel_prefix_document = True

# Suppress specific non-critical warnings
suppress_warnings = [
    "sphinx_autodoc_typehints.forward_reference",
    "sphinx_autodoc_typehints.guarded_import",
    "design.grid",
    "ref.python",
    "autosectionlabel.*",
    "docutils",
    "myst.header",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_logo = "_static/images/logos/anomalib-icon.png"
html_favicon = "_static/images/logos/anomalib-favicon.png"
html_static_path = ["_static"]
html_theme_options = {
    "logo": {
        "text": "Anomalib",
    },
}

# Add references to example files
html_context = {"examples_path": str(examples_path)}

# External documentation references
intersphinx_mapping: dict[str, tuple[str, None]] = {}

autodoc_warningiserror = False


class _DummyDirective(Directive):
    """Ignore unsupported status directives found in third-party docstrings."""

    has_content = True

    @staticmethod
    def run() -> list[Any]:
        """Return no nodes for the unsupported directive."""
        return []


directives.register_directive("betastatus", _DummyDirective)
