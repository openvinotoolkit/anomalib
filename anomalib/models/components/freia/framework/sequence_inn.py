"""Sequence INN."""

# Copyright (c) 2018-2022 Lynton Ardizzone, Visual Learning Lab Heidelberg.
# SPDX-License-Identifier: MIT
#

# pylint: disable=invalid-name
# flake8: noqa
# pylint: skip-file
# type: ignore
# pydocstyle: noqa

from typing import Iterable, List, Tuple

import torch
from torch import Tensor, nn

from anomalib.models.components.freia.modules.base import InvertibleModule


class SequenceINN(InvertibleModule):
    """Simpler than FrEIA.framework.GraphINN.

    Only supports a sequential series of modules (no splitting, merging,
    branching off).
    Has an append() method, to add new blocks in a more simple way than the
    computation-graph based approach of GraphINN. For example:
    .. code-block:: python
       inn = SequenceINN(channels, dims_H, dims_W)
       for i in range(n_blocks):
           inn.append(FrEIA.modules.AllInOneBlock, clamp=2.0, permute_soft=True)
       inn.append(FrEIA.modules.HaarDownsampling)
       # and so on
    """

    def __init__(self, *dims: int, force_tuple_output=False):
        super().__init__([dims])

        self.shapes = [tuple(dims)]
        self.conditions = []
        self.module_list = nn.ModuleList()

        self.force_tuple_output = force_tuple_output

    def append(self, module_class, cond=None, cond_shape=None, **kwargs):
        """Append a reversible block from FrEIA.modules to the network.

        Args:
          module_class: Class from FrEIA.modules.
          cond (int): index of which condition to use (conditions will be passed as list to forward()).
            Conditioning nodes are not needed for SequenceINN.
          cond_shape (tuple[int]): the shape of the condition tensor.
          **kwargs: Further keyword arguments that are passed to the constructor of module_class (see example).
        """

        dims_in = [self.shapes[-1]]
        self.conditions.append(cond)

        if cond is not None:
            kwargs["dims_c"] = [cond_shape]

        module = module_class(dims_in, **kwargs)
        self.module_list.append(module)
        ouput_dims = module.output_dims(dims_in)
        assert len(ouput_dims) == 1, "Module has more than one output"
        self.shapes.append(ouput_dims[0])

    def __getitem__(self, item):
        """Get item."""
        return self.module_list.__getitem__(item)

    def __len__(self):
        """Get length."""
        return self.module_list.__len__()

    def __iter__(self):
        """Iter."""
        return self.module_list.__iter__()

    def output_dims(self, input_dims: List[Tuple[int]]) -> List[Tuple[int]]:
        """Output Dims."""
        if not self.force_tuple_output:
            raise ValueError(
                "You can only call output_dims on a SequentialINN " "when setting force_tuple_output=True."
            )
        return input_dims

    def forward(
        self, x_or_z: Tensor, c: Iterable[Tensor] = None, rev: bool = False, jac: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Execute the sequential INN in forward or inverse (rev=True) direction.

        Args:
            x_or_z: input tensor (in contrast to GraphINN, a list of
                    tensors is not supported, as SequenceINN only has
                    one input).
            c: list of conditions.
            rev: whether to compute the network forward or reversed.
            jac: whether to compute the log jacobian
        Returns:
            z_or_x (Tensor): network output.
            jac (Tensor): log-jacobian-determinant.
        """

        iterator = range(len(self.module_list))
        log_det_jac = 0

        if rev:
            iterator = reversed(iterator)

        if torch.is_tensor(x_or_z):
            x_or_z = (x_or_z,)
        for i in iterator:
            if self.conditions[i] is None:
                x_or_z, j = self.module_list[i](x_or_z, jac=jac, rev=rev)
            else:
                x_or_z, j = self.module_list[i](x_or_z, c=[c[self.conditions[i]]], jac=jac, rev=rev)
            log_det_jac = j + log_det_jac

        return x_or_z if self.force_tuple_output else x_or_z[0], log_det_jac
