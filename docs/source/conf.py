# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import anomalib  # noqa

# -- Project information -----------------------------------------------------

PROJECT = "Anomalib"
COPYRIGHT = "2021, Anomalib Contributors"
AUTHOR = "Anomalib Contributors"
VERSION = anomalib.__version__

html_title = " ".join((PROJECT, COPYRIGHT, "documentation"))

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinxemoji.sphinxemoji",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
    "autoapi.extension",
]

autoapi_dirs = ["../../anomalib"]
autoapi_root = "api"
autoapi_type = "python"

autosummary_generate = True
autodoc_member_order = "groupwise"
autoclass_content = "both"

autodoc_default_options = {
    "members": True,
    "methods": True,
    "special-members": "__call__",
    "exclude-members": "_abc_impl",
    "show-inheritance": True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "alabaster"
html_theme = "furo"

# Sphinx will add “permalinks” for each heading and description environment as paragraph signs that
#  become visible when the mouse hovers over them.
# This value determines the text for the permalink; it defaults to "¶". Set it to None or the empty
#  string to disable permalinks.
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_add_permalinks
html_permalinks = True
html_permalinks_icon = "¶"

# True to prefix each section label with the name of the document it is in, followed by a colon.
#  For example, index:Introduction for a section called Introduction that appears in document index.rst.
#  Useful for avoiding ambiguity when the same section heading appears in different documents.
# http://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
