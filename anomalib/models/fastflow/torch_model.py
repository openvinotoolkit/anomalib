"""FastFlow Torch Model Implementation."""

# Original Code
# Copyright (c) 2022 @gathierry
# https://github.com/gathierry/FastFlow/.
# SPDX-License-Identifier: Apache-2.0
#
# Modified
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Callable, List, Tuple, Union

import timm
import torch
from timm.models.cait import Cait
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor, nn

from anomalib.models.components.freia.framework import SequenceINN
from anomalib.models.components.freia.modules import AllInOneBlock
from anomalib.models.fastflow.anomaly_map import AnomalyMapGenerator


def subnet_conv_func(kernel_size: int, hidden_ratio: float) -> Callable:
    """Subnet Convolutional Function.

    Callable class or function ``f``, called as ``f(channels_in, channels_out)`` and
        should return a torch.nn.Module.
        Predicts coupling coefficients :math:`s, t`.

    Args:
        kernel_size (int): Kernel Size
        hidden_ratio (float): Hidden ratio to compute number of hidden channels.

    Returns:
        Callable: Sequential for the subnet constructor.
    """

    def subnet_conv(in_channels: int, out_channels: int) -> nn.Sequential:
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv


def create_fast_flow_block(
    input_dimensions: List[int],
    conv3x3_only: bool,
    hidden_ratio: float,
    flow_steps: int,
    clamp: float = 2.0,
) -> SequenceINN:
    """Create NF Fast Flow Block.

    This is to create Normalizing Flow (NF) Fast Flow model block based on
    Figure 2 and Section 3.3 in the paper.

    Args:
        input_dimensions (List[int]): Input dimensions (Channel, Height, Width)
        conv3x3_only (bool): Boolean whether to use conv3x3 only or conv3x3 and conv1x1.
        hidden_ratio (float): Ratio for the hidden layer channels.
        flow_steps (int): Flow steps.
        clamp (float, optional): Clamp. Defaults to 2.0.

    Returns:
        SequenceINN: FastFlow Block.
    """
    nodes = SequenceINN(*input_dimensions)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class FastflowLoss(nn.Module):
    """FastFlow Loss."""

    def forward(self, hidden_variables: List[Tensor], jacobians: List[Tensor]) -> Tensor:
        """Calculate the Fastflow loss.

        Args:
            hidden_variables (List[Tensor]): Hidden variables from the fastflow model. f: X -> Z
            jacobians (List[Tensor]): Log of the jacobian determinants from the fastflow model.

        Returns:
            Tensor: _description_
        """
        loss = torch.tensor(0.0, device=hidden_variables[0].device)
        for (hidden_variable, jacobian) in zip(hidden_variables, jacobians):
            loss += torch.mean(0.5 * torch.sum(hidden_variable**2, dim=(1, 2, 3)) - jacobian)
        return loss


class FastflowModel(nn.Module):
    """FastFlow.

    Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows.

    Args:
        input_size (Tuple[int, int]): Model input size.
        backbone (str): Backbone CNN network
        flow_steps (int): Flow steps.
        conv3x3_only (bool, optinoal): Use only conv3x3 in fast_flow model. Defaults to False.
        hidden_ratio (float, optional): Ratio to calculate hidden var channels. Defaults to 1.0.

    Raises:
        ValueError: When the backbone is not supported.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        backbone: str,
        flow_steps: int,
        conv3x3_only: bool = False,
        hidden_ratio: float = 1.0,
    ) -> None:
        super().__init__()

        self.input_size = input_size

        if backbone in ["cait_m48_448", "deit_base_distilled_patch16_384"]:
            self.feature_extractor = timm.create_model(backbone, pretrained=True)
            channels = [768]
            scales = [16]
        elif backbone in ["resnet18", "wide_resnet50_2"]:
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3],
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # for transformers, use their pretrained norm w/o grad
            # for resnets, self.norms are trainable LayerNorm
            self.norms = nn.ModuleList()
            for channel, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [channel, int(input_size[0] / scale), int(input_size[1] / scale)],
                        elementwise_affine=True,
                    )
                )
        else:
            raise ValueError(
                f"Backbone {backbone} is not supported. List of available backbones are "
                "[cait_m48_448, deit_base_distilled_patch16_384, resnet18, wide_resnet50_2]."
            )

        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        self.fast_flow_blocks = nn.ModuleList()
        for channel, scale in zip(channels, scales):
            self.fast_flow_blocks.append(
                create_fast_flow_block(
                    input_dimensions=[channel, int(input_size[0] / scale), int(input_size[1] / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

    def forward(self, x: Tensor) -> Union[Tuple[List[Tensor], List[Tensor]], Tensor]:
        """Forward-Pass the input to the FastFlow Model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Union[Tuple[Tensor, Tensor], Tensor]: During training, return
                (hidden_variables, log-of-the-jacobian-determinants).
                During the validation/test, return the anomaly map.
        """
        # pylint: disable=invalid-name

        return_val: Union[Tuple[List[Tensor], List[Tensor]], Tensor]

        self.feature_extractor.eval()
        if isinstance(self.feature_extractor, VisionTransformer):
            x = self.feature_extractor.patch_embed(x)
            cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
            if self.feature_extractor.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat(
                    (
                        cls_token,
                        self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1,
                )
            x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
            for i in range(8):  # paper Table 6. Block Index = 7
                x = self.feature_extractor.blocks[i](x)
            x = self.feature_extractor.norm(x)
            x = x[:, 2:, :]
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size[0] // 16, self.input_size[1] // 16)
            features = [x]
        elif isinstance(self.feature_extractor, Cait):
            x = self.feature_extractor.patch_embed(x)
            x = x + self.feature_extractor.pos_embed
            x = self.feature_extractor.pos_drop(x)
            for i in range(41):  # paper Table 6. Block Index = 40
                x = self.feature_extractor.blocks[i](x)
            N, _, C = x.shape
            x = self.feature_extractor.norm(x)
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size[0] // 16, self.input_size[1] // 16)
            features = [x]
        else:
            features = self.feature_extractor(x)
            features = [self.norms[i](feature) for i, feature in enumerate(features)]

        # Compute the hidden variable f: X -> Z and log-likelihood of the jacobian
        # (See Section 3.3 in the paper.)
        # NOTE: output variable has z, and jacobian tuple for each fast-flow blocks.
        hidden_variables: List[Tensor] = []
        log_jacobians: List[Tensor] = []
        for fast_flow_block, feature in zip(self.fast_flow_blocks, features):
            hidden_variable, log_jacobian = fast_flow_block(feature)
            hidden_variables.append(hidden_variable)
            log_jacobians.append(log_jacobian)

        return_val = (hidden_variables, log_jacobians)

        if not self.training:
            return_val = self.anomaly_map_generator(hidden_variables)

        return return_val
