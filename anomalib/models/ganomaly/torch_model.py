"""Torch models defining encoder, decoder, Generator and Discriminator.

Code adapted from https://github.com/samet-akcay/ganomaly.
"""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.


import math

from torch import Tensor, nn


class Encoder(nn.Module):
    """Encoder Network.

    Args:
        input_size (int): Size of input image
        latent_vec_size (int): Size of latent vector z
        num_input_channels (int): Number of input channels in the image
        n_features (int): Number of features per convolution layer
        extra_layers (int): Number of extra layers since the network uses only a single encoder layer by default.
            Defaults to 0.
    """

    def __init__(
        self,
        input_size: int,
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
    ):
        super().__init__()

        assert input_size % 16 == 0, "Input size should be a multiple of 16"  # Why?

        self.input_layers = nn.Sequential()

        self.input_layers.add_module(
            f"initial-conv-{num_input_channels}-{n_features}",
            nn.Conv2d(num_input_channels, n_features, kernel_size=4, stride=2, padding=1, bias=False),
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
        input_size = input_size // 2
        while input_size > 4:
            in_features = n_features
            out_features = n_features * 2
            self.pyramid_features.add_module(
                f"pyramid-{in_features}-{out_features}-conv",
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False),
            )
            self.pyramid_features.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            self.pyramid_features.add_module(f"pyramid-{out_features}-relu", nn.LeakyReLU(0.2, inplace=True))
            n_features = out_features
            input_size = input_size // 2

        # Final conv
        if add_final_conv_layer:
            self.final_conv_layer = nn.Conv2d(
                n_features, latent_vec_size, kernel_size=4, stride=1, padding=0, bias=False
            )

    def forward(self, input_tensor: Tensor):
        """Return latent vectors."""

        output = self.input_layers(input_tensor)
        if len(self.extra_layers) > 0:
            output = self.extra_layers(output)
        output = self.pyramid_features(output)
        if self.final_conv_layer is not None:
            output = self.final_conv_layer(output)

        return output


class Decoder(nn.Module):
    """Decoder Network.

    Args:
        input_size (int): Size of input image
        latent_vec_size (int): Size of latent vector z
        num_input_channels (int): Number of input channels in the image
        n_features (int): Number of features per convolution layer
        extra_layers (int): Number of extra layers since the network uses only a single encoder layer by default.
            Defaults to 0.
    """

    def __init__(
        self, input_size: int, latent_vec_size: int, num_input_channels: int, n_features: int, extra_layers: int = 0
    ):
        super().__init__()
        assert input_size % 16 == 0, "Input size should be a multiple of 16"  # Why again?

        self.latent_input = nn.Sequential()

        # Calculate input channel size to recreate inverse pyramid
        exp_factor = int(math.log(input_size // 4, 2)) - 1
        n_input_features = n_features * (2 ** exp_factor)

        # CNN layer for latent vector input
        self.latent_input.add_module(
            f"initial-{latent_vec_size}-{n_input_features}-convt",
            nn.ConvTranspose2d(latent_vec_size, n_input_features, kernel_size=4, stride=1, padding=0, bias=False),
        )
        self.latent_input.add_module(f"initial-{n_input_features}-batchnorm", nn.BatchNorm2d(n_input_features))
        self.latent_input.add_module(f"initial-{n_input_features}-relu", nn.ReLU(True))

        # Create inverse pyramid
        self.inverse_pyramid = nn.Sequential()
        input_size = input_size // 2
        while input_size > 4:
            in_features = n_input_features
            out_features = n_input_features // 2
            self.inverse_pyramid.add_module(
                f"pyramid-{in_features}-{out_features}-convt",
                nn.ConvTranspose2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False),
            )
            self.inverse_pyramid.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            self.inverse_pyramid.add_module(f"pyramid-{out_features}-relu", nn.ReLU(True))
            n_input_features = out_features
            input_size = input_size // 2

        # Extra Layers
        self.extra_layers = nn.Sequential()
        for layer in range(extra_layers):
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-conv",
                nn.Conv2d(n_input_features, n_input_features, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-batchnorm", nn.BatchNorm2d(n_input_features)
            )
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-relu", nn.LeakyReLU(0.2, inplace=True)
            )

        # Final layers
        self.final_layers = nn.Sequential()
        self.final_layers.add_module(
            f"final-{n_input_features}-{num_input_channels}-convt",
            nn.ConvTranspose2d(n_input_features, num_input_channels, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.final_layers.add_module(f"final-{num_input_channels}-tanh", nn.Tanh())

    def forward(self, input_tensor):
        """Return generated image."""
        output = self.latent_input(input_tensor)
        output = self.inverse_pyramid(output)
        if len(self.extra_layers) > 0:
            output = self.extra_layers(output)
        output = self.final_layers(output)
        return output


class Discriminator(nn.Module):
    """Discriminator.

        Made of only one encoder layer which takes x and x_hat to produce a score.

    Args:
        input_size (int): Input image size.
        num_input_channels (int): Number of image channels.
        n_features (int): Number of feature maps in each convolution layer.
        extra_layers (int, optional): Add extra intermediate layers. Defaults to 0.
    """

    def __init__(self, input_size: int, num_input_channels: int, n_features: int, extra_layers: int = 0):
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

    def forward(self, input_tensor):
        """Return class of object and features."""
        features = self.features(input_tensor)
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)
        return classifier, features


class Generator(nn.Module):
    """Generator model.

    Made of an encoder-decoder-encoder architecture.

    Args:
        input_size (int): Size of input data.
        latent_vec_size (int): Dimension of latent vector produced between the first encoder-decoder.
        num_input_channels (int): Number of channels in input image.
        n_features (int): Number of feature maps in each convolution layer.
        extra_layers (int, optional): Extra intermediate layers in the encoder/decoder. Defaults to 0.
        add_final_conv_layer (bool, optional): Add a final convolution layer in the decoder. Defaults to True.
    """

    def __init__(
        self,
        input_size: int,
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
    ):
        super().__init__()
        self.encoder1 = Encoder(
            input_size, latent_vec_size, num_input_channels, n_features, extra_layers, add_final_conv_layer
        )
        self.decoder = Decoder(input_size, latent_vec_size, num_input_channels, n_features, extra_layers)
        self.encoder2 = Encoder(
            input_size, latent_vec_size, num_input_channels, n_features, extra_layers, add_final_conv_layer
        )

    def forward(self, input_tensor):
        """Return generated image and the latent vectors."""
        latent_i = self.encoder1(input_tensor)
        gen_image = self.decoder(latent_i)
        latent_o = self.encoder2(gen_image)
        return gen_image, latent_i, latent_o
