"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Project information -----------------------------------------------------
https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""

from __future__ import annotations

import sys
from pathlib import Path

# Define the path to your module using Path
module_path = Path(__file__).parent.parent / "src"

# Insert the path to sys.path
sys.path.insert(0, str(module_path.resolve()))

project = "Anomalib"
copyright = "2023, Intel OpenVINO"  # noqa: A001
author = "Intel OpenVINO"
release = "2022"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx_design",
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

myst_enable_extensions = [
    "colon_fence",
    # other MyST extensions...
]
nbsphinx_allow_errors = True
templates_path = ["_templates"]
exclude_patterns: list[str] = []


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
