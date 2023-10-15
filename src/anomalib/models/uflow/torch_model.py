import torch.nn as nn
from FrEIA import framework as ff, modules as fm

from .feature_extraction import get_feature_extractor
from .anomaly_map import AnomalyMapGenerator


class UflowModel(nn.Module):
    def __init__(
            self,
            input_size=(448, 448),
            flow_steps=4,
            backbone="mcait",
            affine_clamp=2.0,
            affine_subnet_channels_ratio=1.0,
            permute_soft=False
    ):
        super(UflowModel, self).__init__()

        self.input_size = input_size
        self.affine_clamp = affine_clamp
        self.affine_subnet_channels_ratio = affine_subnet_channels_ratio
        self.permute_soft = permute_soft

        self.feature_extractor = get_feature_extractor(backbone, input_size)
        self.flow = self.build_flow(flow_steps)
        self.anomaly_map_generator = AnomalyMapGenerator(input_size)

    def build_flow(self, flow_steps):
        input_nodes = []
        for channel, s_factor in zip(self.feature_extractor.channels, self.feature_extractor.scale_factors):
            input_nodes.append(
                ff.InputNode(channel, self.input_size[0] // s_factor, self.input_size[1] // s_factor, name=f"cond_{channel}")
            )

        nodes, output_nodes = [], []
        last_node = input_nodes[-1]
        for i in reversed(range(1, len(input_nodes))):
            flows = self.get_flow_stage(last_node, flow_steps)
            volume_size = flows[-1].output_dims[0][0]
            split = ff.Node(
                flows[-1], fm.Split,
                {'section_sizes': (volume_size // 8 * 4, volume_size - volume_size // 8 * 4), 'dim': 0},
                name=f'split_{i + 1}'
            )
            output = ff.OutputNode(split.out1, name=f'output_scale_{i + 1}')
            up = ff.Node(split.out0, fm.IRevNetUpsampling, {}, name=f'up_{i + 1}')
            last_node = ff.Node([input_nodes[i - 1].out0, up.out0], fm.Concat, {'dim': 0}, name=f'cat_{i}')

            output_nodes.append(output)
            nodes.extend([*flows, split, up, last_node])

        flows = self.get_flow_stage(last_node, flow_steps)
        output = ff.OutputNode(flows[-1], name='output_scale_1')

        output_nodes.append(output)
        nodes.extend(flows)

        return ff.GraphINN(input_nodes + nodes + output_nodes[::-1])

    def get_flow_stage(self, in_node, flow_steps, condition_node=None):

        def get_affine_coupling_subnet(kernel_size, subnet_channels_ratio):
            def affine_coupling_subnet(in_channels, out_channels):
                mid_channels = int(in_channels * subnet_channels_ratio)
                return nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size, padding="same"),
                    nn.ReLU(),
                    nn.Conv2d(mid_channels, out_channels, kernel_size, padding="same"),
                )

            return affine_coupling_subnet

        def glow_params(k):
            return {
                'subnet_constructor': get_affine_coupling_subnet(k, self.affine_subnet_channels_ratio),
                'affine_clamping': self.affine_clamp,
                'permute_soft': self.permute_soft
            }

        flow_size = in_node.output_dims[0][-1]
        nodes = []
        for step in range(flow_steps):
            nodes.append(ff.Node(
                in_node,
                fm.AllInOneBlock,
                glow_params(3 if step % 2 == 0 else 1),
                conditions=condition_node,
                name=f"flow{flow_size}_step{step}"
            ))
            in_node = nodes[-1]
        return nodes

    def forward(self, image):
        features = self.feature_extractor(image)
        z, ljd = self.encode(features)

        if self.training:
            return z, ljd
        else:
            return self.anomaly_map_generator(z)

    def encode(self, features):
        z, ljd = self.flow(features, rev=False)
        if len(self.feature_extractor.scales) == 1:
            z = [z]
        return z, ljd
