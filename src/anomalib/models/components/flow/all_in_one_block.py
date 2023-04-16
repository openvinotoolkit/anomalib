"""All In One Block Layer."""

# Copyright (c) https://github.com/vislearn/FrEIA
# SPDX-License-Identifier: MIT

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from typing import Callable

import torch
import torch.nn.functional as F
from FrEIA.modules import InvertibleModule
from scipy.stats import special_ortho_group
from torch import Tensor, nn


def _global_scale_sigmoid_activation(input: Tensor) -> Tensor:
    """Global scale sigmoid activation.

    Args:
        input (Tensor): Input tensor

    Returns:
        Tensor: Sigmoid activation
    """
    return 10 * torch.sigmoid(input - 2.0)


def _global_scale_softplus_activation(input: Tensor) -> Tensor:
    """Global scale softplus activation.

    Args:
        input (Tensor): Input tensor

    Returns:
        Tensor: Softplus activation
    """
    softplus = nn.Softplus(beta=0.5)
    return 0.1 * softplus(input)


def _global_scale_exp_activation(input: Tensor) -> Tensor:
    """Global scale exponential activation.

    Args:
        input (Tensor): Input tensor

    Returns:
        Tensor: Exponential activation
    """
    return torch.exp(input)


class AllInOneBlock(InvertibleModule):
    """Module combining the most common operations in a normalizing flow or similar model.

    It combines affine coupling, permutation, and global affine transformation
    ('ActNorm'). It can also be used as GIN coupling block, perform learned
    householder permutations, and use an inverted pre-permutation. The affine
    transformation includes a soft clamping mechanism, first used in Real-NVP.
    The block as a whole performs the following computation:

    .. math::

        y = V\\,R \\; \\Psi(s_\\mathrm{global}) \\odot \\mathrm{Coupling}\\Big(R^{-1} V^{-1} x\\Big)+ t_\\mathrm{global}

    - The inverse pre-permutation of x (i.e. :math:`R^{-1} V^{-1}`) is optional (see
      ``reverse_permutation`` below).
    - The learned householder reflection matrix
      :math:`V` is also optional all together (see ``learned_householder_permutation``
      below).
    - For the coupling, the input is split into :math:`x_1, x_2` along
      the channel dimension. Then the output of the coupling operation is the
      two halves :math:`u = \\mathrm{concat}(u_1, u_2)`.

      .. math::

          u_1 &= x_1 \\odot \\exp \\Big( \\alpha \\; \\mathrm{tanh}\\big( s(x_2) \\big)\\Big) + t(x_2) \\\\
          u_2 &= x_2

      Because :math:`\\mathrm{tanh}(s) \\in [-1, 1]`, this clamping mechanism prevents
      exploding values in the exponential. The hyperparameter :math:`\\alpha` can be adjusted.

    """

    def __init__(
        self,
        dims_in,
        dims_c=[],
        subnet_constructor: Callable | None = None,
        affine_clamping: float = 2.0,
        gin_block: bool = False,
        global_affine_init: float = 1.0,
        global_affine_type: str = "SOFTPLUS",
        permute_soft: bool = False,
        learned_householder_permutation: int = 0,
        reverse_permutation: bool = False,
    ):
        """
        Args:
          subnet_constructor:
            class or callable ``f``, called as ``f(channels_in, channels_out)`` and
            should return a torch.nn.Module. Predicts coupling coefficients :math:`s, t`.
          affine_clamping:
            clamp the output of the multiplicative coefficients before
            exponentiation to +/- ``affine_clamping`` (see :math:`\\alpha` above).
          gin_block:
            Turn the block into a GIN block from Sorrenson et al, 2019.
            Makes it so that the coupling operations as a whole is volume preserving.
          global_affine_init:
            Initial value for the global affine scaling :math:`s_\mathrm{global}`.
          global_affine_init:
            ``'SIGMOID'``, ``'SOFTPLUS'``, or ``'EXP'``. Defines the activation to be used
            on the beta for the global affine scaling (:math:`\\Psi` above).
          permute_soft:
            bool, whether to sample the permutation matrix :math:`R` from :math:`SO(N)`,
            or to use hard permutations instead. Note, ``permute_soft=True`` is very slow
            when working with >512 dimensions.
          learned_householder_permutation:
            Int, if >0, turn on the matrix :math:`V` above, that represents
            multiple learned householder reflections. Slow if large number.
            Dubious whether it actually helps network performance.
          reverse_permutation:
            Reverse the permutation before the block, as introduced by Putzky
            et al, 2019. Turns on the :math:`R^{-1} V^{-1}` pre-multiplication above.
        """

        super().__init__(dims_in, dims_c)

        channels = dims_in[0][0]
        # rank of the tensors means 1d, 2d, 3d tensor etc.
        self.input_rank = len(dims_in[0]) - 1
        # tuple containing all dims except for batch-dim (used at various points)
        self.sum_dims = tuple(range(1, 2 + self.input_rank))

        if len(dims_c) == 0:
            self.conditional = False
            self.condition_channels = 0
        else:
            assert tuple(dims_c[0][1:]) == tuple(
                dims_in[0][1:]
            ), f"Dimensions of input and condition don't agree: {dims_c} vs {dims_in}."
            self.conditional = True
            self.condition_channels = sum(dc[0] for dc in dims_c)

        split_len1 = channels - channels // 2
        split_len2 = channels // 2
        self.splits = [split_len1, split_len2]

        try:
            self.permute_function = {0: F.linear, 1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[self.input_rank]
        except KeyError:
            raise ValueError(f"Data is {1 + self.input_rank}D. Must be 1D-4D.")

        self.in_channels = channels
        self.clamp = affine_clamping
        self.GIN = gin_block
        self.reverse_pre_permute = reverse_permutation
        self.householder = learned_householder_permutation

        if permute_soft and channels > 512:
            warnings.warn(
                (
                    "Soft permutation will take a very long time to initialize "
                    f"with {channels} feature channels. Consider using hard permutation instead."
                )
            )

        # global_scale is used as the initial value for the global affine scale
        # (pre-activation). It is computed such that
        # global_scale_activation(global_scale) = global_affine_init
        # the 'magic numbers' (specifically for sigmoid) scale the activation to
        # a sensible range.
        if global_affine_type == "SIGMOID":
            global_scale = 2.0 - torch.log(torch.tensor([10.0 / global_affine_init - 1.0]))
            self.global_scale_activation = _global_scale_sigmoid_activation
        elif global_affine_type == "SOFTPLUS":
            global_scale = 2.0 * torch.log(torch.exp(torch.tensor(0.5 * 10.0 * global_affine_init)) - 1)
            self.global_scale_activation = _global_scale_softplus_activation
        elif global_affine_type == "EXP":
            global_scale = torch.log(torch.tensor(global_affine_init))
            self.global_scale_activation = _global_scale_exp_activation
        else:
            raise ValueError('Global affine activation must be "SIGMOID", "SOFTPLUS" or "EXP"')

        self.global_scale = nn.Parameter(torch.ones(1, self.in_channels, *([1] * self.input_rank)) * global_scale)
        self.global_offset = nn.Parameter(torch.zeros(1, self.in_channels, *([1] * self.input_rank)))

        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            indices = torch.randperm(channels)
            w = torch.zeros((channels, channels))
            w[torch.arange(channels), indices] = 1.0

        if self.householder:
            # instead of just the permutation matrix w, the learned housholder
            # permutation keeps track of reflection vectors vk, in addition to a
            # random initial permutation w_0.
            self.vk_householder = nn.Parameter(0.2 * torch.randn(self.householder, channels), requires_grad=True)
            self.w_perm = None
            self.w_perm_inv = None
            self.w_0 = nn.Parameter(torch.FloatTensor(w), requires_grad=False)
        else:
            self.w_perm = nn.Parameter(
                torch.FloatTensor(w).view(channels, channels, *([1] * self.input_rank)), requires_grad=False
            )
            self.w_perm_inv = nn.Parameter(
                torch.FloatTensor(w.T).view(channels, channels, *([1] * self.input_rank)), requires_grad=False
            )

        if subnet_constructor is None:
            raise ValueError("Please supply a callable subnet_constructor" "function or object (see docstring)")
        self.subnet = subnet_constructor(self.splits[0] + self.condition_channels, 2 * self.splits[1])
        self.last_jac = None

    def _construct_householder_permutation(self):
        """Computes a permutation matrix from the reflection vectors that are
        learned internally as nn.Parameters."""
        w = self.w_0
        for vk in self.vk_householder:
            w = torch.mm(w, torch.eye(self.in_channels).to(w.device) - 2 * torch.ger(vk, vk) / torch.dot(vk, vk))

        for i in range(self.input_rank):
            w = w.unsqueeze(-1)
        return w

    def _permute(self, x, rev=False):
        """Performs the permutation and scaling after the coupling operation.
        Returns transformed outputs and the LogJacDet of the scaling operation."""
        if self.GIN:
            scale = 1.0
            perm_log_jac = 0.0
        else:
            scale = self.global_scale_activation(self.global_scale)
            perm_log_jac = torch.sum(torch.log(scale))

        if rev:
            return ((self.permute_function(x, self.w_perm_inv) - self.global_offset) / scale, perm_log_jac)
        else:
            return (self.permute_function(x * scale + self.global_offset, self.w_perm), perm_log_jac)

    def _pre_permute(self, x, rev=False):
        """Permutes before the coupling block, only used if
        reverse_permutation is set"""
        if rev:
            return self.permute_function(x, self.w_perm)
        else:
            return self.permute_function(x, self.w_perm_inv)

    def _affine(self, x, a, rev=False):
        """Given the passive half, and the pre-activation outputs of the
        coupling subnetwork, perform the affine coupling operation.
        Returns both the transformed inputs and the LogJacDet."""

        # the entire coupling coefficient tensor is scaled down by a
        # factor of ten for stability and easier initialization.
        a *= 0.1
        ch = x.shape[1]

        sub_jac = self.clamp * torch.tanh(a[:, :ch])
        if self.GIN:
            sub_jac -= torch.mean(sub_jac, dim=self.sum_dims, keepdim=True)

        if not rev:
            return (x * torch.exp(sub_jac) + a[:, ch:], torch.sum(sub_jac, dim=self.sum_dims))
        else:
            return ((x - a[:, ch:]) * torch.exp(-sub_jac), -torch.sum(sub_jac, dim=self.sum_dims))

    def forward(self, x, c=[], rev=False, jac=True):
        """See base class docstring"""
        if self.householder:
            self.w_perm = self._construct_householder_permutation()
            if rev or self.reverse_pre_permute:
                self.w_perm_inv = self.w_perm.transpose(0, 1).contiguous()

        if rev:
            x, global_scaling_jac = self._permute(x[0], rev=True)
            x = (x,)
        elif self.reverse_pre_permute:
            x = (self._pre_permute(x[0], rev=False),)

        x1, x2 = torch.split(x[0], self.splits, dim=1)

        if self.conditional:
            x1c = torch.cat([x1, *c], 1)
        else:
            x1c = x1

        if not rev:
            a1 = self.subnet(x1c)
            x2, j2 = self._affine(x2, a1)
        else:
            a1 = self.subnet(x1c)
            x2, j2 = self._affine(x2, a1, rev=True)

        log_jac_det = j2
        x_out = torch.cat((x1, x2), 1)

        if not rev:
            x_out, global_scaling_jac = self._permute(x_out, rev=False)
        elif self.reverse_pre_permute:
            x_out = self._pre_permute(x_out, rev=True)

        # add the global scaling Jacobian to the total.
        # trick to get the total number of non-channel dimensions:
        # number of elements of the first channel of the first batch member
        n_pixels = x_out[0, :1].numel()
        log_jac_det += (-1) ** rev * n_pixels * global_scaling_jac

        return (x_out,), log_jac_det

    def output_dims(self, input_dims):
        return input_dims
