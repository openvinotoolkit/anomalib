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
