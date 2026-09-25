# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GRD-Net residual networks."""

import torch

from anomalib.models.image.draem.torch_model import DiscriminativeSubNetwork
from anomalib.models.image.grdnet.torch_model import (
    GRDNetDiscriminator,
    GRDNetGenerator,
    GRDNetModel,
    ResidualEncoder,
)


def test_encoder_stage_shapes() -> None:
    """The canonical encoder should reduce 128-pixel tiles to 512 by 8 by 8 features."""
    encoder = ResidualEncoder()
    stage_outputs: list[torch.Size] = []
    hooks = [
        encoder.blocks[index].register_forward_hook(lambda _module, _inputs, output: stage_outputs.append(output.shape))
        for index in (1, 3, 5, 7)
    ]

    with torch.no_grad():
        output = encoder(torch.rand((1, 3, 128, 128)))
    for hook in hooks:
        hook.remove()

    assert stage_outputs == [
        torch.Size((1, 64, 64, 64)),
        torch.Size((1, 128, 32, 32)),
        torch.Size((1, 256, 16, 16)),
        torch.Size((1, 512, 8, 8)),
    ]
    assert output.shape == (1, 512, 8, 8)


def test_generator_shapes_range_and_parameter_independence() -> None:
    """The generator should use independent encoders and 32-channel latent maps."""
    generator = GRDNetGenerator().eval()

    with torch.no_grad():
        latent, reconstruction, reconstruction_latent = generator(torch.rand((1, 3, 128, 128)))

    assert latent.shape == (1, 32, 8, 8)
    assert reconstruction.shape == (1, 3, 128, 128)
    assert reconstruction_latent.shape == (1, 32, 8, 8)
    assert torch.all((reconstruction >= 0.0) & (reconstruction <= 1.0))
    encoder_storage = {parameter.data_ptr() for parameter in generator.encoder.parameters()}
    reconstruction_encoder_storage = {
        parameter.data_ptr() for parameter in generator.reconstruction_encoder.parameters()
    }
    projection_storage = {parameter.data_ptr() for parameter in generator.latent_projection.parameters()}
    reconstruction_projection_storage = {
        parameter.data_ptr() for parameter in generator.reconstruction_projection.parameters()
    }
    assert encoder_storage.isdisjoint(reconstruction_encoder_storage)
    assert projection_storage.isdisjoint(reconstruction_projection_storage)


def test_discriminator_shapes() -> None:
    """The discriminator should return final residual features and one logit per tile."""
    discriminator = GRDNetDiscriminator().eval()

    with torch.no_grad():
        features, logits = discriminator(torch.rand((2, 3, 128, 128)))

    assert features.shape == (2, 512, 8, 8)
    assert logits.shape == (2, 1)


def test_generator_has_finite_gradients() -> None:
    """Both generator encoders and the reconstruction path should receive finite gradients."""
    generator = GRDNetGenerator(base_features=4, latent_channels=2)
    latent, reconstruction, reconstruction_latent = generator(torch.rand((2, 3, 32, 32)))

    (latent.mean() + reconstruction.mean() + reconstruction_latent.mean()).backward()

    gradients = [parameter.grad for parameter in generator.parameters() if parameter.requires_grad]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_discriminator_has_finite_gradients() -> None:
    """The adversarial encoder and classifier should receive finite gradients."""
    discriminator = GRDNetDiscriminator(input_size=(32, 32), base_features=4)
    features, logits = discriminator(torch.rand((2, 3, 32, 32)))

    (features.mean() + logits.mean()).backward()

    gradients = [parameter.grad for parameter in discriminator.parameters() if parameter.requires_grad]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_model_composes_draem_segmentator() -> None:
    """GRD-Net should directly reuse DRÆM's six-channel discriminative network."""
    model = GRDNetModel(input_size=(32, 32), base_features=4, latent_channels=2).eval()

    with torch.no_grad():
        logits = model.segmentator(torch.rand((1, 6, 64, 64)))

    assert type(model.segmentator) is DiscriminativeSubNetwork
    assert logits.shape == (1, 2, 64, 64)
