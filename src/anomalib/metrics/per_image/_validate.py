"""Utils for validating arguments and results.

`torch` is imported in the functions that use it, so this module can be used in numpy-standalone mode.
"""

from __future__ import annotations

from typing import Any


def is_tensor(tensor: Any, argname: str | None = None) -> None:  # noqa: ANN401
    """Validate that `tensor` is a `torch.Tensor`."""
    from torch import Tensor

    argname = f"'{argname}'" if argname is not None else "argument"

    if not isinstance(tensor, Tensor):
        msg = f"Expected {argname} to be a tensor, but got {type(tensor)}"
        raise TypeError(msg)


def num_threshs(num_threshs: int) -> None:
    """Validate that `num_threshs` is a positive integer >= 2."""
    if not isinstance(num_threshs, int):
        msg = f"Expected `num_threshs` to be an integer, but got {type(num_threshs)}"
        raise TypeError(msg)

    if num_threshs < 2:
        msg = f"If argument `num_threshs` is an integer, expected it to be larger than 1, but got {num_threshs}"
        raise ValueError(msg)


def same_shape(*args) -> None:
    """Works both for tensors and ndarrays."""
    assert len(args) > 0
    shapes = sorted({tuple(arg.shape) for arg in args})
    if len(shapes) > 1:
        msg = f"Expected arguments to have the same shape, but got {shapes}"
        raise ValueError(msg)


def rate(rate: float | int, zero_ok: bool, one_ok: bool) -> None:
    """Validates a rate parameter.

    Args:
        rate (float | int): The rate to be validated.
        zero_ok (bool): Flag indicating if rate can be 0.
        one_ok (bool): Flag indicating if rate can be 1.
    """
    if not isinstance(rate, float | int):
        msg = f"Expected rate to be a float or int, but got {type(rate)}."
        raise TypeError(msg)

    if rate < 0.0 or rate > 1.0:
        msg = f"Rate `{rate}` is not a valid because it must be in [0, 1]."
        raise ValueError(msg)

    if not zero_ok and rate == 0.0:
        msg = "Rate cannot be 0."
        raise ValueError(msg)

    if not one_ok and rate == 1.0:
        msg = "Rate cannot be 1."
        raise ValueError(msg)


def rate_range(bounds: tuple[float, float]) -> None:
    """Validates the range of rates within `bounds`.

    Args:
        bounds (tuple[float, float]): The lower and upper bounds of the rates.
    """
    if not isinstance(bounds, tuple):
        msg = f"Expected `bounds` to be a tuple, but got {type(bounds)}"
        raise TypeError(msg)

    if len(bounds) != 2:
        msg = f"Expected `bounds` to be a tuple of length 2, but got {len(bounds)}"
        raise ValueError(msg)

    lower, upper = bounds
    rate(lower, zero_ok=False, one_ok=False)
    rate(upper, zero_ok=False, one_ok=True)

    if lower >= upper:
        msg = f"Expected `bounds[1]` > `bounds[0]`, but got {bounds[1]} <= {bounds[0]}"
        raise ValueError(msg)
