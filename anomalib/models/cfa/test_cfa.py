"""Test CFA Model Implementation."""

import pytest
import torch
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader
from utils.cfa import Descriptor as OldDescriptor

from anomalib.models.cfa.cnn.resnet import wide_resnet50_2
from anomalib.models.cfa.datasets.mvtec import MVTecDataset
from anomalib.models.cfa.torch_model import CfaModel
from anomalib.models.cfa.torch_model import CoordConv2d as NewCoordConv2d
from anomalib.models.cfa.torch_model import Descriptor as NewDescriptor
from anomalib.models.cfa.torch_model import get_feature_extractor
from anomalib.models.cfa.utils.cfa import DSVDD
from anomalib.models.cfa.utils.coordconv import CoordConv2d as OldCoordConv2d


def initialize_weights(m) -> None:
    torch.manual_seed(0)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


# @pytest.mark.parametrize("input", [torch.rand((4, 1792, 56, 56)), torch.rand((4, 1792, 56, 56)).cuda()])
# def test_coord_conv2d_layer(input) -> None:

#     use_cuda = True if input.device.type == "cuda" else False

#     old_coord_conv2d = OldCoordConv2d(1792, 1792, 1, with_r=False, use_cuda=use_cuda).to(input.device)
#     new_coord_conv2d = NewCoordConv2d(1792, 1792, 1).to(input.device)

#     old_coord_conv2d.apply(initialize_weights)
#     new_coord_conv2d.apply(initialize_weights)

#     assert torch.allclose(old_coord_conv2d(input), new_coord_conv2d(input))


# @pytest.mark.parametrize(
#     "features",
#     [
#         # Create a feature map from Wide-ReNet50-2
#         [torch.rand(4, 256, 56, 56).cuda(), torch.rand(4, 512, 28, 28).cuda(), torch.rand(4, 1024, 14, 14).cuda()],
#     ],
# )
# def test_descriptor(features) -> None:
#     """Test Descriptors."""

#     # Old Descriptor always use cuda upon availability. The new one can be run on cpu.
#     old_descriptor = OldDescriptor(gamma_d=1, cnn="wide_resnet50_2").to(features[0].device)
#     new_descriptor = NewDescriptor(gamma_d=1, backbone="wide_resnet50_2").to(features[0].device)

#     old_descriptor.apply(initialize_weights)
#     new_descriptor.apply(initialize_weights)

#     old_target_oriented_features = old_descriptor(features)
#     new_target_oriented_features = new_descriptor(features)

#     assert torch.allclose(old_target_oriented_features, new_target_oriented_features)


class TestCfaModel:
    def test_cfa_model(self) -> None:
        """Test Init Centroid should return the same memory bank."""
        device = torch.device("cuda")
        train_dataset = MVTecDataset("/home/sakcay/projects/anomalib/datasets/MVTec/", "zipper")
        train_loader = DataLoader(train_dataset, 4)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Compare the feature extractor
        _, (input, _, _) = next(enumerate(train_loader))
        input = input.cuda()

        old_feature_extractor = wide_resnet50_2(pretrained=True, progress=True).to(device)
        old_feature_extractor.eval()

        new_feature_extractor = get_feature_extractor("wide_resnet50_2", device=torch.device("cuda"))

        old_features = old_feature_extractor(input)
        new_features = new_feature_extractor(input)
        new_features = [val for val in new_features.values()]

        for old_feature, new_feature in zip(old_features, new_features):
            assert torch.allclose(old_feature, new_feature, atol=1e-1), "Old and new features should match."
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Compute the memory bank.
        cfa_old = DSVDD(old_feature_extractor, train_loader, "wide_resnet50_2", 1, 1, device).to(device)
        cfa_new = CfaModel(new_feature_extractor, train_loader, "wide_resnet50_2", 1, 1, device).to(device)
        assert torch.allclose(cfa_old.C, cfa_new.memory_bank, atol=1e-1)
        assert cfa_new.memory_bank.requires_grad is False
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Compute Forward-Pass.
        old_loss, _ = cfa_old(old_features)
        new_loss, _ = cfa_new(new_features)
        assert (old_loss - new_loss).abs() / 1000 < 1e-1, "Old and new losses should match."
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # Compute Loss.
        # features = torch.rand((4, 1792, 56, 56)).cuda()
        # features = rearrange(features, "b c h w -> b (h w) c")

        # old_loss = cfa_old._soft_boundary(features)
        # new_loss = cfa_new.compute_loss(features)
        # # Divide by 1000 because the loss is multiplied by 1000 in the implementation.
        # assert (old_loss - new_loss).abs()/1000 < 1e-1
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
