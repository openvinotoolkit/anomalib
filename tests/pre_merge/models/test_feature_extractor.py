import pytest
import torch
from torchvision.models.efficientnet import EfficientNet_B5_Weights

from anomalib.models.components.feature_extractors import (
    TimmFeatureExtractor,
    get_torchfx_feature_extractor,
)


class TestFeatureExtractor:
    @pytest.mark.parametrize(
        "backbone",
        ["resnet18", "wide_resnet50_2"],
    )
    @pytest.mark.parametrize(
        "pretrained",
        [True, False],
    )
    def test_timm_feature_extraction(self, backbone, pretrained):
        layers = ["layer1", "layer2", "layer3"]
        model = TimmFeatureExtractor(backbone=backbone, layers=layers, pre_trained=pretrained)
        test_input = torch.rand((32, 3, 256, 256))
        features = model(test_input)

        if backbone == "resnet18":
            assert features["layer1"].shape == torch.Size((32, 64, 64, 64))
            assert features["layer2"].shape == torch.Size((32, 128, 32, 32))
            assert features["layer3"].shape == torch.Size((32, 256, 16, 16))
            assert model.out_dims == [64, 128, 256]
            assert model.idx == [1, 2, 3]
        elif backbone == "wide_resnet50_2":
            assert features["layer1"].shape == torch.Size((32, 256, 64, 64))
            assert features["layer2"].shape == torch.Size((32, 512, 32, 32))
            assert features["layer3"].shape == torch.Size((32, 1024, 16, 16))
            assert model.out_dims == [256, 512, 1024]
            assert model.idx == [1, 2, 3]
        else:
            pass

    def test_torchfx_feature_extraction(self):
        model = get_torchfx_feature_extractor("resnet18", ["layer1", "layer2", "layer3"])
        test_input = torch.rand((32, 3, 256, 256))
        features = model(test_input)
        assert features["layer1"].shape == torch.Size((32, 64, 64, 64))
        assert features["layer2"].shape == torch.Size((32, 128, 32, 32))
        assert features["layer3"].shape == torch.Size((32, 256, 16, 16))

        model = get_torchfx_feature_extractor(
            backbone="efficientnet_b5", return_nodes=["features.6.8"], weights=EfficientNet_B5_Weights.DEFAULT
        )
        features = model(test_input)
        assert features["features.6.8"].shape == torch.Size((32, 304, 8, 8))
