"""Base Module."""

# Copyright (c) 2018-2022 Lynton Ardizzone, Visual Learning Lab Heidelberg.
# SPDX-License-Identifier: MIT
#

# flake8: noqa
# pylint: skip-file
# type: ignore
# pydocstyle: noqa

from typing import Iterable, List, Tuple

import torch.nn as nn
from torch import Tensor


class InvertibleModule(nn.Module):
    r"""Base class for all invertible modules in FrEIA.

    Given ``module``, an instance of some InvertibleModule.
    This ``module`` shall be invertible in its input dimensions,
    so that the input can be recovered by applying the module
    in backwards mode (``rev=True``), not to be confused with
    ``pytorch.backward()`` which computes the gradient of an operation::
        x = torch.randn(BATCH_SIZE, DIM_COUNT)
        c = torch.randn(BATCH_SIZE, CONDITION_DIM)
        # Forward mode
        z, jac = module([x], [c], jac=True)
        # Backward mode
        x_rev, jac_rev = module(z, [c], rev=True)
    The ``module`` returns :math:`\\log \\det J = \\log \\left| \\det \\frac{\\partial f}{\\partial x} \\right|`
    of the operation in forward mode, and
    :math:`-\\log | \\det J | = \\log \\left| \\det \\frac{\\partial f^{-1}}{\\partial z} \\right| = -\\log \\left| \\det \\frac{\\partial f}{\\partial x} \\right|`
    in backward mode (``rev=True``).
    Then, ``torch.allclose(x, x_rev) == True`` and ``torch.allclose(jac, -jac_rev) == True``.
    """

    def __init__(self, dims_in: Iterable[Tuple[int]], dims_c: Iterable[Tuple[int]] = None):
        """Initialize.

        Args:
            dims_in: list of tuples specifying the shape of the inputs to this
                     operator: ``dims_in = [shape_x_0, shape_x_1, ...]``
            dims_c:  list of tuples specifying the shape of the conditions to
                     this operator.
        """
        super().__init__()
        if dims_c is None:
            dims_c = []
        self.dims_in = list(dims_in)
        self.dims_c = list(dims_c)

    def forward(
        self, x_or_z: Iterable[Tensor], c: Iterable[Tensor] = None, rev: bool = False, jac: bool = True
    ) -> Tuple[Tuple[Tensor], Tensor]:
        r"""Forward/Backward Pass.

        Perform a forward (default, ``rev=False``) or backward pass (``rev=True``) through this module/operator.

        **Note to implementers:**
        - Subclasses MUST return a Jacobian when ``jac=True``, but CAN return a
          valid Jacobian when ``jac=False`` (not punished). The latter is only recommended
          if the computation of the Jacobian is trivial.
        - Subclasses MUST follow the convention that the returned Jacobian be
          consistent with the evaluation direction. Let's make this more precise:
          Let :math:`f` be the function that the subclass represents. Then:
          .. math::
              J &= \\log \\det \\frac{\\partial f}{\\partial x} \\\\
              -J &= \\log \\det \\frac{\\partial f^{-1}}{\\partial z}.
          Any subclass MUST return :math:`J` for forward evaluation (``rev=False``),
          and :math:`-J` for backward evaluation (``rev=True``).

        Args:
            x_or_z: input data (array-like of one or more tensors)
            c:      conditioning data (array-like of none or more tensors)
            rev:    perform backward pass
            jac:    return Jacobian associated to the direction
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not provide forward(...) method")

    def log_jacobian(self, *args, **kwargs):
        """This method is deprecated, and does nothing except raise a warning."""
        raise DeprecationWarning(
            "module.log_jacobian(...) is deprecated. "
            "module.forward(..., jac=True) returns a "
            "tuple (out, jacobian) now."
        )

    def output_dims(self, input_dims: List[Tuple[int]]) -> List[Tuple[int]]:
        """Use for shape inference during construction of the graph.

        MUST be implemented for each subclass of ``InvertibleModule``.

        Args:
          input_dims: A list with one entry for each input to the module.
            Even if the module only has one input, must be a list with one
            entry. Each entry is a tuple giving the shape of that input,
            excluding the batch dimension. For example for a module with one
            input, which receives a 32x32 pixel RGB image, ``input_dims`` would
            be ``[(3, 32, 32)]``

        Returns:
            A list structured in the same way as ``input_dims``. Each entry
            represents one output of the module, and the entry is a tuple giving
            the shape of that output. For example if the module splits the image
            into a right and a left half, the return value should be
            ``[(3, 16, 32), (3, 16, 32)]``. It is up to the implementor of the
            subclass to ensure that the total number of elements in all inputs
            and all outputs is consistent.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not provide output_dims(...)")
