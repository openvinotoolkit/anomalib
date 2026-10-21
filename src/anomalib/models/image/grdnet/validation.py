# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Constructor argument validation for GRD-Net."""

import math


def _validate_texture_source(texture_source: str) -> None:
    """Validate the public texture-source option."""
    if texture_source not in {"random", "image"}:
        msg = f"Unknown texture source {texture_source!r}. Expected 'random' or 'image'."
        raise ValueError(msg)


def _validate_numeric(
    name: str,
    value: float,
    minimum: float,
    maximum: float | None = None,
    *,
    inclusive_minimum: bool = True,
) -> float:
    """Validate one finite public numeric argument."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        msg = f"{name} must be a finite number, got {value!r}."
        raise ValueError(msg)
    if (inclusive_minimum and value < minimum) or (not inclusive_minimum and value <= minimum):
        relation = "at least" if inclusive_minimum else "greater than"
        msg = f"{name} must be {relation} {minimum}, got {value!r}."
        raise ValueError(msg)
    if maximum is not None and value > maximum:
        msg = f"{name} must be at most {maximum}, got {value!r}."
        raise ValueError(msg)
    return float(value)


def _validate_integer(name: str, value: int, minimum: int) -> int:
    """Validate one public integer argument without accepting booleans or floats."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        msg = f"{name} must be an integer of at least {minimum}, got {value!r}."
        raise ValueError(msg)
    return value
