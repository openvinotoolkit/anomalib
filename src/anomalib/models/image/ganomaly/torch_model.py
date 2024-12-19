"""Torch models defining encoder, decoder, generator and discriminator networks.

The GANomaly model consists of several key components:

1. Encoder: Compresses input images into latent vectors
2. Decoder: Reconstructs images from latent vectors
3. Generator: Combines encoder-decoder-encoder for image generation
4. Discriminator: Distinguishes real from generated images

The architecture follows an encoder-decoder-encoder pattern where:
- First encoder compresses input image to latent space
- Decoder reconstructs the image from latent vector
- Second encoder re-encodes reconstructed image
- Anomaly score is based on difference between latent vectors

Example:
    >>> from anomalib.models.image.ganomaly.torch_model import GanomalyModel
    >>> model = GanomalyModel(
    ...     input_size=(256, 256),
    ...     num_input_channels=3,
    ...     n_features=64,
    ...     latent_vec_size=100,
    ...     extra_layers=0,
    ...     add_final_conv_layer=True
    ... )
    >>> input_tensor = torch.randn(32, 3, 256, 256)
    >>> output = model(input_tensor)

Code adapted from:
    Title: GANomaly - PyTorch Implementation
    Authors: Samet Akcay
    URL: https://github.com/samet-akcay/ganomaly
    License: MIT

See Also:
    - :class:`anomalib.models.image.ganomaly.lightning_model.Ganomaly`:
        Lightning implementation of the GANomaly model
    - :class:`anomalib.models.image.ganomaly.loss.GeneratorLoss`:
        Loss function for the generator network
    - :class:`anomalib.models.image.ganomaly.loss.DiscriminatorLoss`:
        Loss function for the discriminator network
"""

# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.data.utils.image import pad_nextpow2


class Encoder(nn.Module):
    """Encoder Network.

    Compresses input images into latent vectors through a series of convolution
    layers.

    Args:
        input_size (tuple[int, int]): Size of input image (height, width)
        latent_vec_size (int): Size of output latent vector
        num_input_channels (int): Number of input image channels
        n_features (int): Number of feature maps in convolution layers
        extra_layers (int, optional): Number of extra intermediate layers.
            Defaults to ``0``.
        add_final_conv_layer (bool, optional): Whether to add final convolution
            layer. Defaults to ``True``.

    Example:
        >>> encoder = Encoder(
        ...     input_size=(256, 256),
        ...     latent_vec_size=100,
        ...     num_input_channels=3,
        ...     n_features=64
        ... )
        >>> input_tensor = torch.randn(32, 3, 256, 256)
        >>> latent = encoder(input_tensor)
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
    ) -> None:
        super().__init__()

        self.input_layers = nn.Sequential()
        self.input_layers.add_module(
            f"initial-conv-{num_input_channels}-{n_features}",
            nn.Conv2d(num_input_channels, n_features, kernel_size=4, stride=2, padding=4, bias=False),
        )
        self.input_layers.add_module(f"initial-relu-{n_features}", nn.LeakyReLU(0.2, inplace=True))

        # Extra Layers
        self.extra_layers = nn.Sequential()

        for layer in range(extra_layers):
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_features}-conv",
                nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.extra_layers.add_module(f"extra-layers-{layer}-{n_features}-batchnorm", nn.BatchNorm2d(n_features))
            self.extra_layers.add_module(f"extra-layers-{layer}-{n_features}-relu", nn.LeakyReLU(0.2, inplace=True))

        # Create pyramid features to reach latent vector
        self.pyramid_features = nn.Sequential()
        pyramid_dim = min(*input_size) // 2  # Use the smaller dimension to create pyramid.
        while pyramid_dim > 4:
            in_features = n_features
            out_features = n_features * 2
            self.pyramid_features.add_module(
                f"pyramid-{in_features}-{out_features}-conv",
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False),
            )
            self.pyramid_features.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            self.pyramid_features.add_module(f"pyramid-{out_features}-relu", nn.LeakyReLU(0.2, inplace=True))
            n_features = out_features
            pyramid_dim = pyramid_dim // 2

        # Final conv
        if add_final_conv_layer:
            self.final_conv_layer = nn.Conv2d(
                n_features,
                latent_vec_size,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder network.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape
                ``(batch_size, channels, height, width)``

        Returns:
            torch.Tensor: Latent vector tensor
        """
        output = self.input_layers(input_tensor)
        output = self.extra_layers(output)
        output = self.pyramid_features(output)
        if self.final_conv_layer is not None:
            output = self.final_conv_layer(output)

        return output


class Decoder(nn.Module):
    """Decoder Network.

    Reconstructs images from latent vectors through transposed convolutions.

    Args:
        input_size (tuple[int, int]): Size of output image (height, width)
        latent_vec_size (int): Size of input latent vector
        num_input_channels (int): Number of output image channels
        n_features (int): Number of feature maps in convolution layers
        extra_layers (int, optional): Number of extra intermediate layers.
            Defaults to ``0``.

    Example:
        >>> decoder = Decoder(
        ...     input_size=(256, 256),
        ...     latent_vec_size=100,
        ...     num_input_channels=3,
        ...     n_features=64
        ... )
        >>> latent = torch.randn(32, 100, 1, 1)
        >>> reconstruction = decoder(latent)
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
    ) -> None:
        super().__init__()

        self.latent_input = nn.Sequential()

        # Calculate input channel size to recreate inverse pyramid
        exp_factor = math.ceil(math.log(min(input_size) // 2, 2)) - 2
        n_input_features = n_features * (2**exp_factor)

        # CNN layer for latent vector input
        self.latent_input.add_module(
            f"initial-{latent_vec_size}-{n_input_features}-convt",
            nn.ConvTranspose2d(
                latent_vec_size,
                n_input_features,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
        )
        self.latent_input.add_module(f"initial-{n_input_features}-batchnorm", nn.BatchNorm2d(n_input_features))
        self.latent_input.add_module(f"initial-{n_input_features}-relu", nn.ReLU(inplace=True))

        # Create inverse pyramid
        self.inverse_pyramid = nn.Sequential()
        pyramid_dim = min(*input_size) // 2  # Use the smaller dimension to create pyramid.
        while pyramid_dim > 4:
            in_features = n_input_features
            out_features = n_input_features // 2
            self.inverse_pyramid.add_module(
                f"pyramid-{in_features}-{out_features}-convt",
                nn.ConvTranspose2d(
                    in_features,
                    out_features,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
            )
            self.inverse_pyramid.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            self.inverse_pyramid.add_module(f"pyramid-{out_features}-relu", nn.ReLU(inplace=True))
            n_input_features = out_features
            pyramid_dim = pyramid_dim // 2

        # Extra Layers
        self.extra_layers = nn.Sequential()
        for layer in range(extra_layers):
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-conv",
                nn.Conv2d(n_input_features, n_input_features, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-batchnorm",
                nn.BatchNorm2d(n_input_features),
            )
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-relu",
                nn.LeakyReLU(0.2, inplace=True),
            )

        # Final layers
        self.final_layers = nn.Sequential()
        self.final_layers.add_module(
            f"final-{n_input_features}-{num_input_channels}-convt",
            nn.ConvTranspose2d(
                n_input_features,
                num_input_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
        )
        self.final_layers.add_module(f"final-{num_input_channels}-tanh", nn.Tanh())

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder network.

        Args:
            input_tensor (torch.Tensor): Input latent tensor

        Returns:
            torch.Tensor: Reconstructed image tensor
        """
        output = self.latent_input(input_tensor)
        output = self.inverse_pyramid(output)
        output = self.extra_layers(output)
        return self.final_layers(output)


class Discriminator(nn.Module):
    """Discriminator Network.

    Classifies images as real or generated using a modified encoder architecture.

    Args:
        input_size (tuple[int, int]): Input image size (height, width)
        num_input_channels (int): Number of input image channels
        n_features (int): Number of feature maps in convolution layers
        extra_layers (int, optional): Number of extra intermediate layers.
            Defaults to ``0``.

    Example:
        >>> discriminator = Discriminator(
        ...     input_size=(256, 256),
        ...     num_input_channels=3,
        ...     n_features=64
        ... )
        >>> input_tensor = torch.randn(32, 3, 256, 256)
        >>> prediction, features = discriminator(input_tensor)
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
    ) -> None:
        super().__init__()
        encoder = Encoder(input_size, 1, num_input_channels, n_features, extra_layers)
        layers = []
        for block in encoder.children():
            if isinstance(block, nn.Sequential):
                layers.extend(list(block.children()))
            else:
                layers.append(block)

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through discriminator network.

        Args:
            input_tensor (torch.Tensor): Input image tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - Classification scores (real/fake)
                - Intermediate features
        """
        features = self.features(input_tensor)
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)
        return classifier, features


class Generator(nn.Module):
    """Generator Network.

    Combines encoder-decoder-encoder architecture for image generation and
    reconstruction.

    Args:
        input_size (tuple[int, int]): Input/output image size (height, width)
        latent_vec_size (int): Size of latent vector between encoder-decoder
        num_input_channels (int): Number of input/output image channels
        n_features (int): Number of feature maps in convolution layers
        extra_layers (int, optional): Number of extra intermediate layers.
            Defaults to ``0``.
        add_final_conv_layer (bool, optional): Add final convolution to encoders.
            Defaults to ``True``.

    Example:
        >>> generator = Generator(
        ...     input_size=(256, 256),
        ...     latent_vec_size=100,
        ...     num_input_channels=3,
        ...     n_features=64
        ... )
        >>> input_tensor = torch.randn(32, 3, 256, 256)
        >>> gen_img, latent_i, latent_o = generator(input_tensor)
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
    ) -> None:
        super().__init__()
        self.encoder1 = Encoder(
            input_size,
            latent_vec_size,
            num_input_channels,
            n_features,
            extra_layers,
            add_final_conv_layer,
        )
        self.decoder = Decoder(input_size, latent_vec_size, num_input_channels, n_features, extra_layers)
        self.encoder2 = Encoder(
            input_size,
            latent_vec_size,
            num_input_channels,
            n_features,
            extra_layers,
            add_final_conv_layer,
        )

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through generator network.

        Args:
            input_tensor (torch.Tensor): Input image tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing:
                - Generated image
                - First encoder's latent vector
                - Second encoder's latent vector
        """
        latent_i = self.encoder1(input_tensor)
        gen_image = self.decoder(latent_i)
        latent_o = self.encoder2(gen_image)
        return gen_image, latent_i, latent_o


class GanomalyModel(nn.Module):
    """GANomaly model for anomaly detection.

    Complete model combining Generator and Discriminator networks.

    Args:
        input_size (tuple[int, int]): Input image size (height, width)
        num_input_channels (int): Number of input image channels
        n_features (int): Number of feature maps in convolution layers
        latent_vec_size (int): Size of latent vector between encoder-decoder
        extra_layers (int, optional): Number of extra intermediate layers.
            Defaults to ``0``.
        add_final_conv_layer (bool, optional): Add final convolution to encoders.
            Defaults to ``True``.

    Example:
        >>> model = GanomalyModel(
        ...     input_size=(256, 256),
        ...     num_input_channels=3,
        ...     n_features=64,
        ...     latent_vec_size=100
        ... )
        >>> input_tensor = torch.randn(32, 3, 256, 256)
        >>> output = model(input_tensor)

    References:
        - Title: GANomaly: Semi-Supervised Anomaly Detection via Adversarial
                Training
        - Authors: Samet Akcay, Amir Atapour-Abarghouei, Toby P. Breckon
        - URL: https://arxiv.org/abs/1805.06725
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        num_input_channels: int,
        n_features: int,
        latent_vec_size: int,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
    ) -> None:
        super().__init__()
        self.generator: Generator = Generator(
            input_size=input_size,
            latent_vec_size=latent_vec_size,
            num_input_channels=num_input_channels,
            n_features=n_features,
            extra_layers=extra_layers,
            add_final_conv_layer=add_final_conv_layer,
        )
        self.discriminator: Discriminator = Discriminator(
            input_size=input_size,
            num_input_channels=num_input_channels,
            n_features=n_features,
            extra_layers=extra_layers,
        )
        self.weights_init(self.generator)
        self.weights_init(self.discriminator)

    @staticmethod
    def weights_init(module: nn.Module) -> None:
        """Initialize DCGAN weights.

        Args:
            module (nn.Module): Neural network module to initialize
        """
        classname = module.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def forward(
        self,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | InferenceBatch:
        """Forward pass through GANomaly model.

        Args:
            batch (torch.Tensor): Batch of input images

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] |
            InferenceBatch:
                If training:
                    - Padded input batch
                    - Generated images
                    - First encoder's latent vectors
                    - Second encoder's latent vectors
                If inference:
                    - Batch containing anomaly scores
        """
        padded_batch = pad_nextpow2(batch)
        fake, latent_i, latent_o = self.generator(padded_batch)
        if self.training:
            return padded_batch, fake, latent_i, latent_o
        scores = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).view(-1)  # convert nx1x1 to n
        return InferenceBatch(pred_score=scores)
