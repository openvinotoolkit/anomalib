"""UFlow Anomaly Map Generator Implementation."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F  # noqa: N812
from omegaconf import ListConfig
from torch import nn

from .nfa_tree import NFATree, compute_number_of_tests


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap and segmentation."""

    def __init__(self, input_size: ListConfig | tuple) -> None:
        super().__init__()
        self.input_size = input_size if isinstance(input_size, tuple) else tuple(input_size)

    def forward(self, latent_variables: list[torch.Tensor]) -> torch.Tensor:
        """Return anomaly map."""
        return self.compute_anomaly_map(latent_variables)

    def compute_anomaly_map(self, latent_variables: list[torch.Tensor]) -> torch.Tensor:
        """Generate a likelihood-based anomaly map, from latent variables.

        Args:
            latent_variables: List of latent variables from the UFlow model. Each element is a tensor of shape
            (N, Cl, Hl, Wl), where N is the batch size, Cl is the number of channels, and Hl and Wl are the height and
            width of the latent variables, respectively, for each scale l.

        Returns:
            Final Anomaly Map. Tensor of shape (N, 1, H, W), where N is the batch size, and H and W are the height and
            width of the input image, respectively.
        """
        likelihoods = []
        for z in latent_variables:
            # Mean prob by scale. Likelihood is actually with sum instead of mean. Using mean to avoid numerical issues.
            # Also, this way all scales have the same weight, and it does not depend on the number of channels
            log_prob_i = -torch.mean(z**2, dim=1, keepdim=True) * 0.5
            prob_i = torch.exp(log_prob_i)
            likelihoods.append(
                F.interpolate(
                    prob_i,
                    size=self.input_size,
                    mode="bilinear",
                    align_corners=False,
                ),
            )
        return 1 - torch.mean(torch.stack(likelihoods, dim=-1), dim=-1)

    def compute_anomaly_mask(
        self,
        z: list[torch.Tensor],
        log_nfa_threshold: float = 0,
        target_size: tuple[int, int] | int | None = None,
        upsample_mode: str = "bilinear",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This method is not used in the basic functionality of training and testing.

        It is included as it is part of the U-Flow paper, and can be called
        separately if an unsupervised anomaly segmentation is needed.

        Generate an anomaly mask, from latent variables. It is based on the NFA
        (Number of False Alarms) method, which is a statistical method to detect
        anomalies. The NFA is computed as the log of the probability of the null
        hypothesis, which is that all pixels are normal. First, we create a level
        set tree, and compute the Probability of False Alarms (PFA) for all nodes.
        Then we apply a pruning and merging process to the tree, to remove nodes
        with high PFA values. The final tree is used to compute the log-NFA,
        which is finally used to obtain the segmentation mask with an automatic
        threshold.

        Args:
            z (list[torch.Tensor]): List of latent variables from the UFlow model.
                Each element is a tensor of shape (N, Cl, Hl, Wl), where N is the
                batch size, Cl is the number of channels, and Hl and Wl are the height
                and width of the latent variables, respectively, for each scale l.
            log_nfa_threshold (float): Log-NFA threshold for the anomaly mask.
                Defaults to 0.
            target_size (tuple[int, int] | int): Target size for the output anomaly mask.
                Defaults to self.input_size.
            upsample_mode (str): Upsample mode for merging results at different scales.
                Defaults to 'bilinear'.

        Returns:
            detection mask. Tensor of shape (N, 1, target_size, target_size),
                where N is the batch size.
            Anomaly score based on the log-NFA value (-log NFA). Same shape as
                the detection mask.
        """
        if target_size is None:
            target_size = self.input_size

        # NFA by region (Tree)
        log_prob_list: list[torch.Tensor] = []
        for img_idx in range(z[0].shape[0]):
            log_prob_s = []
            for zi in z:
                nfa_tree = NFATree(zi[img_idx])
                log_prob_s.append(nfa_tree.compute_log_prob_map())
            log_prob_list.append(
                torch.cat(
                    [
                        F.interpolate(
                            log_prob_s_i.nan_to_num(0).unsqueeze(0).unsqueeze(0),
                            size=target_size,
                            mode=upsample_mode,
                            **{"align_corners": False} if "nearest" not in upsample_mode else {},
                        )
                        for log_prob_s_i in log_prob_s
                    ],
                    dim=1,
                ),
            )
        log_prob: torch.Tensor = torch.cat(log_prob_list, dim=0)
        log_prob = log_prob.amin(dim=1, keepdim=True)

        log_n_tests = compute_number_of_tests([int(zi.shape[-1] * zi.shape[-2]) for zi in z])
        log_nfa = log_n_tests + log_prob

        detection = log_nfa < log_nfa_threshold

        return detection, -log_nfa
