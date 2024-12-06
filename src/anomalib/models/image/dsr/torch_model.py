"""PyTorch model for the DSR model implementation."""

# Original Code
# Copyright (c) 2022 VitjanZ
# https://github.com/VitjanZ/DSR_anomaly_detection.
# SPDX-License-Identifier: Apache-2.0
#
# Modified
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class DsrModel(nn.Module):
    """DSR PyTorch model.

    Consists of the discrete latent model, image reconstruction network,
    subspace restriction modules, anomaly detection module and upsampling module.

    Args:
        embedding_dim (int): Dimension of codebook embeddings.
        num_embeddings (int): Number of embeddings.
        latent_anomaly_strength (float): Strength of the generated anomalies in the latent space.
        num_hiddens (int): Number of output channels in residual layers.
        num_residual_layers (int): Number of residual layers.
        num_residual_hiddens (int): Number of intermediate channels.
    """

    def __init__(
        self,
        latent_anomaly_strength: float = 0.2,
        embedding_dim: int = 128,
        num_embeddings: int = 4096,
        num_hiddens: int = 128,
        num_residual_layers: int = 2,
        num_residual_hiddens: int = 64,
    ) -> None:
        super().__init__()

        self.image_dim: int = 3
        self.anomaly_map_dim: int = 2
        self.latent_anomaly_strength: float = latent_anomaly_strength

        self.discrete_latent_model = DiscreteLatentModel(
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

        self.image_reconstruction_network = ImageReconstructionNetwork(
            in_channels=embedding_dim * 2,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

        self.subspace_restriction_module_lo = SubspaceRestrictionModule(base_width=embedding_dim)
        self.subspace_restriction_module_hi = SubspaceRestrictionModule(base_width=embedding_dim)

        self.anomaly_detection_module = AnomalyDetectionModule(
            in_channels=2 * self.image_dim,
            out_channels=self.anomaly_map_dim,
            base_width=64,
        )

        self.upsampling_module = UpsamplingModule(
            in_channels=(2 * self.image_dim) + self.anomaly_map_dim,
            out_channels=self.anomaly_map_dim,
            base_width=64,
        )

        for parameters in self.discrete_latent_model.parameters():
            parameters.requires_grad = False

    def load_pretrained_discrete_model_weights(self, ckpt: Path, device: torch.device | str | None = None) -> None:
        """Load pre-trained model weights."""
        self.discrete_latent_model.load_state_dict(torch.load(ckpt, map_location=device))

    def forward(
        self,
        batch: torch.Tensor,
        anomaly_map_to_generate: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the anomaly mask from an input image.

        Args:
            batch (torch.Tensor): Batch of input images.
            anomaly_map_to_generate (torch.Tensor | None): anomaly map to use to generate quantized defects.
            If not training phase 2, should be None.

        Returns:
            dict[str, torch.Tensor]:
            If testing:
                - "anomaly_map": Upsampled anomaly map
                - "pred_score": Image score
            If training phase 2:
                - "recon_feat_hi": Reconstructed non-quantized hi features of defect (F~_hi)
                - "recon_feat_lo": Reconstructed non-quantized lo features of defect (F~_lo)
                - "embedding_bot": Quantized features of non defective img (Q_hi)
                - "embedding_top": Quantized features of non defective img (Q_lo)
                - "obj_spec_image": Object-specific-decoded image (I_spc)
                - "anomaly_map": Predicted segmentation mask (M)
                - "true_mask": Resized ground-truth anomaly map (M_gt)
            If training phase 3:
                - "anomaly_map": Reconstructed anomaly map
        """
        outputs: dict[str, torch.Tensor]

        # Generate latent embeddings decoded image via general object decoder
        if anomaly_map_to_generate is None:
            # either evaluating or training phase 3
            with torch.no_grad():
                latent_model_outputs = self.discrete_latent_model(batch)
                gen_image = latent_model_outputs["recon_image"]
                embd_top = latent_model_outputs["quantized_t"]
                embd_bot = latent_model_outputs["quantized_b"]

                # Get embedders from the discrete latent model
                embedder_bot = self.discrete_latent_model.vq_vae_bot
                embedder_top = self.discrete_latent_model.vq_vae_top

                # Copy embeddings in order to input them to the subspace restriction module
                anomaly_embedding_bot_copy = embd_bot.clone()
                anomaly_embedding_top_copy = embd_top.clone()

                # Apply subspace restriction module to copied embeddings
                _, recon_embd_bot = self.subspace_restriction_module_hi(anomaly_embedding_bot_copy, embedder_bot)
                _, recon_embd_top = self.subspace_restriction_module_lo(anomaly_embedding_top_copy, embedder_top)

                # Upscale top (lo) embedding
                up_quantized_recon_t = self.discrete_latent_model.upsample_t(recon_embd_top)

                # Concat embeddings and reconstruct image (object specific decoder)
                quant_join = torch.cat((up_quantized_recon_t, recon_embd_bot), dim=1)
                obj_spec_image = self.image_reconstruction_network(quant_join)

                # Anomaly detection module
                out_mask = self.anomaly_detection_module(obj_spec_image, gen_image)
                out_mask_sm = torch.softmax(out_mask, dim=1)

            # Mask upsampling and score calculation
            upsampled_mask = self.upsampling_module(obj_spec_image, gen_image, out_mask_sm)
            out_mask_sm_up = torch.softmax(upsampled_mask, dim=1)

            # if training phase 3, return upsampled softmax mask
            if self.training:
                outputs = {"anomaly_map": out_mask_sm_up}
            # if testing, extract image score
            else:
                out_mask_averaged = torch.nn.functional.avg_pool2d(
                    out_mask_sm[:, 1:, :, :],
                    21,
                    stride=1,
                    padding=21 // 2,
                ).detach()
                image_score = torch.amax(out_mask_averaged, dim=(2, 3)).squeeze()

                # prevent crash when image_score is a single value (batch size of 1)
                if image_score.size() == torch.Size([]):
                    image_score = image_score.unsqueeze(0)

                out_mask_cv = out_mask_sm_up[:, 1, :, :]

                outputs = {"anomaly_map": out_mask_cv, "pred_score": image_score}

        elif anomaly_map_to_generate is not None and self.training:
            # we are in phase two

            # Generate anomaly strength factors
            anom_str_lo = (
                torch.rand(batch.shape[0]) * (1.0 - self.latent_anomaly_strength) + self.latent_anomaly_strength
            ).cuda()
            anom_str_hi = (
                torch.rand(batch.shape[0]) * (1.0 - self.latent_anomaly_strength) + self.latent_anomaly_strength
            ).cuda()

            # Generate image through general object decoder, and defective & non defective quantized feature maps.
            with torch.no_grad():
                latent_model_outputs = self.discrete_latent_model(
                    batch,
                    anomaly_map_to_generate,
                    anom_str_lo,
                    anom_str_hi,
                )
            gen_image_def = latent_model_outputs["recon_image"]
            true_anomaly_map = latent_model_outputs["anomaly_mask"]
            embd_top = latent_model_outputs["quantized_t"]
            embd_bot = latent_model_outputs["quantized_b"]
            embd_top_def = latent_model_outputs["anomaly_embedding_lo"]
            embd_bot_def = latent_model_outputs["anomaly_embedding_hi"]

            # Restore the features to normality with the Subspace restriction modules
            recon_feat_hi, recon_embeddings_hi = self.subspace_restriction_module_hi(
                embd_bot_def,
                self.discrete_latent_model.vq_vae_bot,
            )
            recon_feat_lo, recon_embeddings_lo = self.subspace_restriction_module_lo(
                embd_top_def,
                self.discrete_latent_model.vq_vae_top,
            )

            # Reconstruct the image from the reconstructed features
            # with the object-specific image reconstruction module
            up_quantized_recon_t = self.discrete_latent_model.upsample_t(recon_embeddings_lo)
            quant_join = torch.cat((up_quantized_recon_t, recon_embeddings_hi), dim=1)
            spec_image_def = self.image_reconstruction_network(quant_join)

            # Generate the anomaly segmentation map
            out_mask = self.anomaly_detection_module(spec_image_def.detach(), gen_image_def.detach())
            out_mask_sm = torch.softmax(out_mask, dim=1)

            # Outputs
            outputs = {
                "recon_feat_hi": recon_feat_hi,
                "recon_feat_lo": recon_feat_lo,
                "embedding_bot": embd_bot,
                "embedding_top": embd_top,
                "obj_spec_image": spec_image_def,
                "anomaly_map": out_mask_sm,
                "true_anomaly_map": true_anomaly_map,
            }
        else:
            msg = "There should not be an anomaly map to generate when not training"
            raise RuntimeError(msg)

        return outputs


class SubspaceRestrictionModule(nn.Module):
    """Subspace Restriction Module.

    Subspace restriction module that restricts the appearance subspace into configurations
    that agree with normal appearances and applies quantization.

    Args:
        base_width (int): Base dimensionality of the layers of the autoencoder.
    """

    def __init__(self, base_width: int) -> None:
        super().__init__()

        self.unet = SubspaceRestrictionNetwork(in_channels=base_width, out_channels=base_width, base_width=base_width)

    def forward(self, batch: torch.Tensor, quantization: Callable) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate the quantized anomaly-free representation of an anomalous image.

        Args:
            batch (torch.Tensor): Batch of input images.
            quantization (function | object): Quantization function.

        Returns:
            Reconstructed batch of non-quantized features and corresponding quantized features.
        """
        batch = self.unet(batch)
        quantized_b = quantization(batch)
        return batch, quantized_b


class SubspaceRestrictionNetwork(nn.Module):
    """Subspace Restriction Network.

    Subspace restriction network that reconstructs the input image into a
    non-quantized configuration that agrees with normal appearances.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        base_width (int): Base dimensionality of the layers of the autoencoder.
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 64, base_width: int = 64) -> None:
        super().__init__()
        self.base_width = base_width
        self.encoder = FeatureEncoder(in_channels, self.base_width)
        self.decoder = FeatureDecoder(self.base_width, out_channels=out_channels)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Reconstruct non-quantized representation from batch.

        Generate non-quantized feature maps from potentially anomalous images, to
        be quantized into non-anomalous quantized representations.

        Args:
            batch (torch.Tensor): Batch of input images.

        Returns:
            Reconstructed non-quantized representation.
        """
        b1, b2, b3 = self.encoder(batch)
        return self.decoder(b1, b2, b3)


class FeatureEncoder(nn.Module):
    """Feature encoder for the subspace restriction network.

    Args:
        in_channels (int): Number of input channels.
        base_width (int): Base dimensionality of the layers of the autoencoder.
    """

    def __init__(self, in_channels: int, base_width: int) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a batch of input features to the latent space.

        Args:
            batch (torch.Tensor): Batch of input images.

        Returns:
        Encoded feature maps.
        """
        b1 = self.block1(batch)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        return b1, b2, b3


class FeatureDecoder(nn.Module):
    """Feature decoder for the subspace restriction network.

    Args:
        base_width (int): Base dimensionality of the layers of the autoencoder.
        out_channels (int): Number of output channels.
    """

    def __init__(self, base_width: int, out_channels: int = 1) -> None:
        super().__init__()

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )

        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, _: torch.Tensor, __: torch.Tensor, b3: torch.Tensor) -> torch.Tensor:
        """Decode a batch of latent features to a non-quantized representation.

        Args:
            _ (torch.Tensor): Top latent feature layer.
            __ (torch.Tensor): Middle latent feature layer.
            b3 (torch.Tensor): Bottom latent feature layer.

        Returns:
            Decoded non-quantized representation.
        """
        up2 = self.up2(b3)
        db2 = self.db2(up2)

        up3 = self.up3(db2)
        db3 = self.db3(up3)

        return self.fin_out(db3)


class Residual(nn.Module):
    """Residual layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_residual_hiddens (int): Number of intermediate channels.
    """

    def __init__(self, in_channels: int, out_channels: int, num_residual_hiddens: int) -> None:
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_residual_hiddens, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute residual layer.

        Args:
            batch (torch.Tensor): Batch of input images.

        Returns:
            Computed feature maps.
        """
        return batch + self._block(batch)


class ResidualStack(nn.Module):
    """Stack of residual layers.

    Args:
        in_channels (int): Number of input channels.
        num_hiddens (int): Number of output channels in residual layers.
        num_residual_layers (int): Number of residual layers.
        num_residual_hiddens (int): Number of intermediate channels.
    """

    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int) -> None:
        super().__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [Residual(in_channels, num_hiddens, num_residual_hiddens) for _ in range(self._num_residual_layers)],
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute residual stack.

        Args:
            batch (torch.Tensor): Batch of input images.

        Returns:
            Computed feature maps.
        """
        for i in range(self._num_residual_layers):
            batch = self._layers[i](batch)
        return F.relu(batch)


class ImageReconstructionNetwork(nn.Module):
    """Image Reconstruction Network.

    Image reconstruction network that reconstructs the image from a quantized
    representation.

    Args:
        in_channels (int): Number of input channels.
        num_hiddens (int): Number of output channels in residual layers.
        num_residual_layers (int): Number of residual layers.
        num_residual_hiddens (int): Number of intermediate channels.
    """

    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int) -> None:
        super().__init__()
        norm_layer = nn.InstanceNorm2d
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            norm_layer(in_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1),
            norm_layer(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=3, padding=1),
            norm_layer(in_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))

        self.pre_vq_conv = nn.Conv2d(in_channels=in_channels * 4, out_channels=64, kernel_size=1, stride=1)

        self.upblock1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)

        self.upblock2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)

        self._conv_1 = nn.Conv2d(in_channels=64, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 2,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reconstructs an image from a quantized representation.

        Args:
            inputs (torch.Tensor): Quantized features.

        Returns:
            Reconstructed image.
        """
        batch = self.block1(inputs)
        batch = self.mp1(batch)
        batch = self.block2(batch)
        batch = self.mp2(batch)
        batch = self.pre_vq_conv(batch)

        batch = self.upblock1(batch)
        batch = F.relu(batch)
        batch = self.upblock2(batch)
        batch = F.relu(batch)
        batch = self._conv_1(batch)

        batch = self._residual_stack(batch)

        batch = self._conv_trans_1(batch)
        batch = F.relu(batch)

        return self._conv_trans_2(batch)


class UnetEncoder(nn.Module):
    """Encoder of the Unet network.

    Args:
        in_channels (int): Number of input channels.
        base_width (int): Base dimensionality of the layers of the autoencoder.
    """

    def __init__(self, in_channels: int, base_width: int) -> None:
        super().__init__()
        norm_layer = nn.InstanceNorm2d
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True),
        )
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes batch of images into a latent representation.

        Args:
            batch (torch.Tensor): Quantized features.

        Returns:
            Latent representations of the input batch.
        """
        b1 = self.block1(batch)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        return b1, b2, b3, b4


class UnetDecoder(nn.Module):
    """Decoder of the Unet network.

    Args:
        base_width (int): Base dimensionality of the layers of the autoencoder.
        out_channels (int): Number of output channels.
    """

    def __init__(self, base_width: int, out_channels: int = 1) -> None:
        super().__init__()
        norm_layer = nn.InstanceNorm2d
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
        )
        # cat with base*4
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * (4 + 4), base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True),
        )
        # cat with base*2
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 2), base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True),
        )
        # cat with base*1
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * (1 + 1), base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True),
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b1: torch.Tensor, b2: torch.Tensor, b3: torch.Tensor, b4: torch.Tensor) -> torch.Tensor:
        """Decodes latent represnetations into an image.

        Args:
            b1 (torch.Tensor): First (top level) quantized feature map.
            b2 (torch.Tensor): Second quantized feature map.
            b3 (torch.Tensor): Third quantized feature map.
            b4 (torch.Tensor): Fourth (bottom level) quantized feature map.

        Returns:
            Reconstructed image.
        """
        up1 = self.up1(b4)
        cat1 = torch.cat((up1, b3), dim=1)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)
        cat2 = torch.cat((up2, b2), dim=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        cat3 = torch.cat((up3, b1), dim=1)
        db3 = self.db3(cat3)

        return self.fin_out(db3)


class UnetModel(nn.Module):
    """Autoencoder model that reconstructs the input image.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        base_width (int): Base dimensionality of the layers of the autoencoder.
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 64, base_width: int = 64) -> None:
        super().__init__()
        self.encoder = UnetEncoder(in_channels, base_width)
        self.decoder = UnetDecoder(base_width, out_channels=out_channels)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Reconstructs an input batch of images.

        Args:
            batch (torch.Tensor): Batch of input images.

        Returns:
            Reconstructed images.
        """
        b1, b2, b3, b4 = self.encoder(batch)
        return self.decoder(b1, b2, b3, b4)


class AnomalyDetectionModule(nn.Module):
    """Anomaly detection module.

    Module that detects the preseßnce of an anomaly by comparing two images reconstructed by
    the object specific decoder and the general object decoder.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        base_width (int): Base dimensionality of the layers of the autoencoder.
    """

    def __init__(self, in_channels: int, out_channels: int, base_width: int) -> None:
        super().__init__()
        self.unet = UnetModel(in_channels, out_channels, base_width)

    def forward(self, batch_real: torch.Tensor, batch_anomaly: torch.Tensor) -> torch.Tensor:
        """Computes the anomaly map over corresponding real and anomalous images.

        Args:
            batch_real (torch.Tensor): Batch of real, non defective images.
            batch_anomaly (torch.Tensor): Batch of potentially anomalous images.

        Returns:
            The anomaly segmentation map.
        """
        img_x = torch.cat((batch_real, batch_anomaly), dim=1)
        return self.unet(img_x)


class UpsamplingModule(nn.Module):
    """Module that upsamples the generated anomaly mask to full resolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        base_width (int): Base dimensionality of the layers of the autoencoder.
    """

    def __init__(self, in_channels: int = 8, out_channels: int = 2, base_width: int = 64) -> None:
        super().__init__()
        self.unet = UnetModel(in_channels, out_channels, base_width)

    def forward(
        self,
        batch_real: torch.Tensor,
        batch_anomaly: torch.Tensor,
        batch_segmentation_map: torch.Tensor,
    ) -> torch.Tensor:
        """Computes upsampled segmentation maps.

        Args:
            batch_real (torch.Tensor): Batch of real, non defective images.
            batch_anomaly (torch.Tensor): Batch of potentially anomalous images.
            batch_segmentation_map (torch.Tensor): Batch of anomaly segmentation maps.

        Returns:
            Upsampled anomaly segmentation maps.
        """
        img_x = torch.cat((batch_real, batch_anomaly, batch_segmentation_map), dim=1)
        return self.unet(img_x)


class VectorQuantizer(nn.Module):
    """Module that quantizes a given feature map using learned quantization codebooks.

    Args:
        num_embeddings (int): Size of embedding codebook.
        embedding_dim (int): Dimension of embeddings.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        # Source for the VectorQuantizer module: https://github.com/zalandoresearch/pytorch-vq-vae
        super().__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()

        # necessary to correctly load the checkpoint file
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

    @property
    def embedding(self) -> torch.Tensor:
        """Return embedding."""
        return self._embedding

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Calculates quantized feature map.

        Args:
            inputs (torch.Tensor): Non-quantized feature maps.

        Returns:
            Quantized feature maps.
        """
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        quantized = inputs + (quantized - inputs).detach()

        # convert quantized from BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous()


class EncoderBot(nn.Module):
    """Encoder module for bottom quantized feature maps.

    Args:
        in_channels (int): Number of input channels.
        num_hiddens (int): Number of hidden channels.
        num_residual_layers (int): Number of residual layers in residual stacks.
        num_residual_hiddens (int): Number of channels in residual layers.
    """

    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int) -> None:
        super().__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_2 = nn.Conv2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Encode inputs to be quantized into the bottom feature map.

        Args:
            batch (torch.Tensor): Batch of input images.

        Returns:
            Encoded feature maps.
        """
        x = self._conv_1(batch)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class EncoderTop(nn.Module):
    """Encoder module for top quantized feature maps.

    Args:
        in_channels (int): Number of input channels.
        num_hiddens (int): Number of hidden channels.
        num_residual_layers (int): Number of residual layers in residual stacks.
        num_residual_hiddens (int): Number of channels in residual layers.
    """

    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int) -> None:
        super().__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens, kernel_size=4, stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Encode inputs to be quantized into the top feature map.

        Args:
            batch (torch.Tensor): Batch of input images.

        Returns:
            Encoded feature maps.
        """
        x = self._conv_1(batch)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        return self._residual_stack(x)


class DecoderBot(nn.Module):
    """General appearance decoder module to reconstruct images while keeping possible anomalies.

    Args:
        in_channels (int): Number of input channels.
        num_hiddens (int): Number of hidden channels.
        num_residual_layers (int): Number of residual layers in residual stack.
        num_residual_hiddens (int): Number of channels in residual layers.
    """

    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int) -> None:
        super().__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 2,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Decode quantized feature maps into an image.

        Args:
            inputs (torch.Tensor): Quantized feature maps.

        Returns:
            Decoded image.
        """
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


class DiscreteLatentModel(nn.Module):
    """Discrete Latent Model.

    Autoencoder quantized model that encodes the input images into quantized feature maps and generates
    a reconstructed image using the general appearance decoder.

    Args:
        num_hiddens (int): Number of hidden channels.
        num_residual_layers (int): Number of residual layers in residual stacks.
        num_residual_hiddens (int): Number of channels in residual layers.
        num_embeddings (int): Size of embedding dictionary.
        embedding_dim (int): Dimension of embeddings.
    """

    def __init__(
        self,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self._encoder_t = EncoderTop(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)

        self._encoder_b = EncoderBot(3, num_hiddens, num_residual_layers, num_residual_hiddens)

        self._pre_vq_conv_bot = nn.Conv2d(
            in_channels=num_hiddens + embedding_dim,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
        )

        self._pre_vq_conv_top = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        self._vq_vae_top = VectorQuantizer(num_embeddings, embedding_dim)

        self._vq_vae_bot = VectorQuantizer(num_embeddings, embedding_dim)

        self._decoder_b = DecoderBot(embedding_dim * 2, num_hiddens, num_residual_layers, num_residual_hiddens)

        self.upsample_t = nn.ConvTranspose2d(embedding_dim, embedding_dim, 4, stride=2, padding=1)

    @property
    def vq_vae_top(self) -> VectorQuantizer:
        """Return ``self._vq_vae_top``."""
        return self._vq_vae_top

    @property
    def vq_vae_bot(self) -> VectorQuantizer:
        """Return ``self._vq_vae_bot``."""
        return self._vq_vae_bot

    @staticmethod
    def generate_fake_anomalies_joined(
        features: torch.Tensor,
        embeddings: torch.Tensor,
        memory_torch_original: torch.Tensor,
        mask: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        """Generate quantized anomalies.

        Args:
            features (torch.Tensor): Features on which the anomalies will be generated.
            embeddings (torch.Tensor): Embeddings to use to generate the anomalies.
            memory_torch_original (torch.Tensor): Weight of embeddings.
            mask (torch.Tensor): Original anomaly mask.
            strength (float): Strength of generated anomaly.

        Returns:
            torch.Tensor: Anomalous embedding.
        """
        random_embeddings = torch.zeros(
            (embeddings.shape[0], embeddings.shape[2] * embeddings.shape[3], memory_torch_original.shape[1]),
        )
        inputs = features.permute(0, 2, 3, 1).contiguous()

        for k in range(embeddings.shape[0]):
            memory_torch = memory_torch_original
            flat_input = inputs[k].view(-1, memory_torch.shape[1])

            distances_b = (
                torch.sum(flat_input**2, dim=1, keepdim=True)
                + torch.sum(memory_torch**2, dim=1)
                - 2 * torch.matmul(flat_input, memory_torch.t())
            )

            percentage_vectors = strength[k]
            topk = max(1, min(int(percentage_vectors * memory_torch.shape[0]) + 1, memory_torch.shape[0] - 1))
            _, topk_indices = torch.topk(distances_b, topk, dim=1, largest=False)
            topk_indices = topk_indices[:, int(memory_torch.shape[0] * 0.05) :]
            topk = topk_indices.shape[1]

            random_indices_hik = torch.randint(topk, size=(topk_indices.shape[0],))
            random_indices_t = topk_indices[torch.arange(random_indices_hik.shape[0]), random_indices_hik]
            random_embeddings[k] = memory_torch[random_indices_t, :]
        random_embeddings = random_embeddings.reshape(
            (random_embeddings.shape[0], embeddings.shape[2], embeddings.shape[3], random_embeddings.shape[2]),
        )
        random_embeddings_tensor = random_embeddings.permute(0, 3, 1, 2).cuda()

        down_ratio_y = int(mask.shape[2] / embeddings.shape[2])
        down_ratio_x = int(mask.shape[3] / embeddings.shape[3])
        anomaly_mask = torch.nn.functional.max_pool2d(mask, (down_ratio_y, down_ratio_x)).float()

        return anomaly_mask * random_embeddings_tensor + (1.0 - anomaly_mask) * embeddings

    def forward(
        self,
        batch: torch.Tensor,
        anomaly_mask: torch.Tensor | None = None,
        anom_str_lo: torch.Tensor | None = None,
        anom_str_hi: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generate quantized feature maps.

        Generates quantized feature maps of batch of input images as well as their
        reconstruction based on the general appearance decoder.

        Args:
            batch (Tensor): Batch of input images.
            anomaly_mask (Tensor | None): Anomaly mask to be used to generate anomalies on
                the quantized feature maps.
            anom_str_lo (torch.Tensor | None): Strength of generated anomaly lo.
            anom_str_hi (torch.Tensor | None): Strength of generated anomaly hi.

        Returns:
            dict[str, torch.Tensor]:
            If generating an anomaly mask:
                - General object decoder-decoded anomalous image
                - Reshaped ground truth anomaly map
                - Non defective quantized lo feature
                - Non defective quantized hi feature
                - Non quantized subspace encoded defective lo feature
                - Non quantized subspace encoded defective hi feature
            Else:
                - General object decoder-decoded image
                - Quantized lo feature
                - Quantized hi feature
        """
        # Encoder Hi
        enc_b = self._encoder_b(batch)

        # Encoder Lo -- F_Lo
        enc_t = self._encoder_t(enc_b)
        zt = self._pre_vq_conv_top(enc_t)

        # Quantize F_Lo with K_Lo
        quantized_t = self._vq_vae_top(zt)

        # Upsample Q_Lo
        up_quantized_t = self.upsample_t(quantized_t)

        # Concatenate and transform the output of Encoder_Hi and upsampled Q_lo -- F_Hi
        feat = torch.cat((enc_b, up_quantized_t), dim=1)
        zb = self._pre_vq_conv_bot(feat)

        # Quantize F_Hi with K_Hi
        quantized_b = self._vq_vae_bot(zb)

        # generate anomalies
        anomaly_embedding_hi = None
        anomaly_embedding_lo = None

        # define outputs
        outputs = {"quantized_b": quantized_b, "quantized_t": quantized_t}

        if anomaly_mask is not None:
            # Generate feature-based anomalies on F_lo
            anomaly_embedding_lo = self.generate_fake_anomalies_joined(
                zt,
                quantized_t,
                self._vq_vae_top.embedding.weight,
                anomaly_mask,
                anom_str_lo,
            )

            up_quantized_t_defect = self.upsample_t(anomaly_embedding_lo)
            feat_defect = torch.cat((enc_b, up_quantized_t_defect), dim=1)
            zb_defect = self._pre_vq_conv_bot(feat_defect)
            quantized_b_defect = self._vq_vae_bot(zb_defect)

            # Generate feature-based anomalies on F_hi
            anomaly_embedding_hi = self.generate_fake_anomalies_joined(
                zb_defect,
                quantized_b_defect,
                self._vq_vae_bot.embedding.weight,
                anomaly_mask,
                anom_str_hi,
            )

            # get anomaly embeddings
            use_both = torch.randint(0, 2, (batch.shape[0], 1, 1, 1)).cuda().float()
            use_lo = torch.randint(0, 2, (batch.shape[0], 1, 1, 1)).cuda().float()
            use_hi = 1 - use_lo

            anomaly_embedding_hi_usebot = self.generate_fake_anomalies_joined(
                zb,
                quantized_b,
                self._vq_vae_bot.embedding.weight,
                anomaly_mask,
                anom_str_hi,
            )

            anomaly_embedding_lo_usebot = quantized_t
            anomaly_embedding_hi_usetop = quantized_b
            anomaly_embedding_lo_usetop = anomaly_embedding_lo
            anomaly_embedding_hi_not_both = use_hi * anomaly_embedding_hi_usebot + use_lo * anomaly_embedding_hi_usetop
            anomaly_embedding_lo_not_both = use_hi * anomaly_embedding_lo_usebot + use_lo * anomaly_embedding_lo_usetop
            anomaly_embedding_hi = (
                (anomaly_embedding_hi * use_both + anomaly_embedding_hi_not_both * (1.0 - use_both)).detach().clone()
            )
            anomaly_embedding_lo = (
                (anomaly_embedding_lo * use_both + anomaly_embedding_lo_not_both * (1.0 - use_both)).detach().clone()
            )

            anomaly_embedding_hi_copy = anomaly_embedding_hi.clone()
            anomaly_embedding_lo_copy = anomaly_embedding_lo.clone()

            # apply the general appearance decoder to the anomaly embeddings
            up_quantized_anomaly_t = self.upsample_t(anomaly_embedding_lo_copy)
            quant_join_anomaly = torch.cat((up_quantized_anomaly_t, anomaly_embedding_hi_copy), dim=1)
            recon_image = self._decoder_b(quant_join_anomaly)

            # Resize the ground truth anomaly map to closely match the augmented features
            down_ratio_x_hi = int(anomaly_mask.shape[3] / quantized_b_defect.shape[3])
            anomaly_mask_hi = torch.nn.functional.max_pool2d(anomaly_mask, (down_ratio_x_hi, down_ratio_x_hi)).float()
            anomaly_mask_hi = torch.nn.functional.interpolate(anomaly_mask_hi, scale_factor=down_ratio_x_hi)
            down_ratio_x_lo = int(anomaly_mask.shape[3] / quantized_t.shape[3])
            anomaly_mask_lo = torch.nn.functional.max_pool2d(anomaly_mask, (down_ratio_x_lo, down_ratio_x_lo)).float()
            anomaly_mask_lo = torch.nn.functional.interpolate(anomaly_mask_lo, scale_factor=down_ratio_x_lo)
            anomaly_mask = anomaly_mask_lo * use_both + (anomaly_mask_lo * use_lo + anomaly_mask_hi * use_hi) * (
                1.0 - use_both
            )

            # reminder : top = lo, bot = hi!
            outputs["recon_image"] = recon_image
            outputs["anomaly_mask"] = anomaly_mask
            outputs["anomaly_embedding_lo"] = anomaly_embedding_lo
            outputs["anomaly_embedding_hi"] = anomaly_embedding_hi

        else:
            # Concatenate Q_Hi and Q_Lo and input it into the General appearance decoder
            quant_join = torch.cat((up_quantized_t, quantized_b), dim=1)
            recon_image = self._decoder_b(quant_join)

            outputs["recon_image"] = recon_image

        return outputs
