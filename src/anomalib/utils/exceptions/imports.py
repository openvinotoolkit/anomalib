"""Import handling utilities."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from importlib import import_module

logger = logging.getLogger(__name__)


def try_import(import_path: str) -> bool:
    """Try to import a module.

    Args:
        import_path (str): The import path of the module.

    Returns:
        bool: True if import succeeds, False otherwise.
    """
    try:
        import_module(import_path)
    except ImportError:
        import_package = import_path.split(".")[0]
        logger.warning(
            f"Could not find {import_package}. To use this feature, ensure that you have {import_package} installed.",
        )
    else:
        return True
    return False
