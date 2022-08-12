import pytest
import torch

from anomalib.models.components.feature_extractors import FeatureExtractor


class TestFeatureExtractor:
    @pytest.mark.parametrize(
        "backbone",
        ["resnet18", "wide_resnet50_2"],
    )
    @pytest.mark.parametrize(
        "pretrained",
        [True, False],
    )
    def test_feature_extraction(self, backbone, pretrained):
        layers = ["layer1", "layer2", "layer3"]
        model = FeatureExtractor(backbone=backbone, layers=layers, pre_trained=pretrained)
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
