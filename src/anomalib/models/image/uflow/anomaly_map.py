"""UFlow Anomaly Map Generator Implementation."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F  # noqa: N812
from mpmath import binomial, mp
from omegaconf import ListConfig
from scipy import integrate
from torch import Tensor, nn

mp.dps = 15  # Set precision for NFA computation (in case of high_precision=True)


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap and segmentation."""

    def __init__(self, input_size: ListConfig | tuple) -> None:
        super().__init__()
        self.input_size = input_size if isinstance(input_size, tuple) else tuple(input_size)

    def forward(self, latent_variables: list[Tensor]) -> Tensor:
        """Return anomaly map."""
        return self.compute_anomaly_map(latent_variables)

    def compute_anomaly_map(self, latent_variables: list[Tensor]) -> Tensor:
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
        window_size: int = 7,
        binomial_probability_thr: float = 0.5,
        high_precision: bool = False,
    ) -> torch.Tensor:
        """This method is not used in the basic functionality of training and testing.

        It is a bit slow, so we decided to
        leave it as an option for the user. It is included as it is part of the U-Flow paper, and can be called
        separately if an unsupervised anomaly segmentation is needed.

        Generate an anomaly mask, from latent variables. It is based on the NFA (Number of False Alarms) method, which
        is a statistical method to detect anomalies. The NFA is computed as the log of the probability of the null
        hypothesis, which is that all pixels are normal. First, we compute a list of  candidate pixels, with
        suspiciously high values of z^2, by applying a binomial test to each pixel, looking at a window  around it.
        Then, to compute the NFA values (actually the log-NFA), we evaluate how probable is that a pixel  belongs to the
        normal distribution. The null-hypothesis is that under normality assumptions, all candidate pixels are uniformly
        distributed. Then, the detection is based on the concentration of candidate pixels.

        Args:
            z (list[torch.Tensor]): List of latent variables from the UFlow model. Each element is a tensor of shape
                (N, Cl, Hl, Wl), where N is the batch size, Cl is the number of channels, and Hl and Wl are the height
                and width of the latent variables, respectively, for each scale l.
            window_size (int): Window size for the binomial test. Defaults to 7.
            binomial_probability_thr (float): Probability threshold for the binomial test. Defaults to 0.5
            high_precision (bool): Whether to use high precision for the binomial test. Defaults to False.

        Returns:
            Anomaly mask. Tensor of shape (N, 1, H, W), where N is the batch size, and H and W are the height and
            width of the input image, respectively.
        """
        log_prob_l = [
            self.binomial_test(zi, window_size / (2**scale), binomial_probability_thr, high_precision)
            for scale, zi in enumerate(z)
        ]

        log_prob_l_up = torch.cat(
            [F.interpolate(lpl, size=self.input_size, mode="bicubic", align_corners=True) for lpl in log_prob_l],
            dim=1,
        )

        log_prob = torch.sum(log_prob_l_up, dim=1, keepdim=True)

        log_number_of_tests = torch.log10(torch.sum(torch.tensor([zi.shape[-2] * zi.shape[-1] for zi in z])))
        log_nfa = log_number_of_tests + log_prob

        anomaly_score = -log_nfa

        return anomaly_score < 0

    @staticmethod
    def binomial_test(
        z: torch.Tensor,
        window_size: int,
        probability_thr: float,
        high_precision: bool = False,
    ) -> torch.Tensor:
        """The binomial test applied to validate or reject the null hypothesis that the pixel is normal.

        The null hypothesis is that the pixel is normal, and the alternative hypothesis is that the pixel is anomalous.
        The binomial test is applied to a window around the pixel, and the number of pixels in the window that ares
        anomalous is compared to the number of pixels that are expected to be anomalous under the null hypothesis.

        Args:
            z: Latent variable from the UFlow model. Tensor of shape (N, Cl, Hl, Wl), where N is the batch size, Cl is
            the number of channels, and Hl and Wl are the height and width of the latent variables, respectively.
            window_size (int): Window size for the binomial test.
            probability_thr: Probability threshold for the binomial test.
            high_precision: Whether to use high precision for the binomial test.

        Returns:
            Log of the probability of the null hypothesis.

        """
        tau = st.chi2.ppf(probability_thr, 1)
        half_win = np.max([int(window_size // 2), 1])

        n_chann = z.shape[1]

        # Candidates
        z2 = F.pad(z**2, tuple(4 * [half_win]), "reflect").detach().cpu()
        z2_unfold_h = z2.unfold(-2, 2 * half_win + 1, 1)
        z2_unfold_hw = z2_unfold_h.unfold(-2, 2 * half_win + 1, 1).numpy()
        observed_candidates_k = np.sum(z2_unfold_hw >= tau, axis=(-2, -1))

        # All volume together
        observed_candidates = np.sum(observed_candidates_k, axis=1, keepdims=True)
        x = observed_candidates / n_chann
        n = int((2 * half_win + 1) ** 2)

        # Low precision
        if not high_precision:
            log_prob = torch.tensor(st.binom.logsf(x, n, 1 - probability_thr) / np.log(10))
        # High precision - good and slow
        else:
            to_mp = np.frompyfunc(mp.mpf, 1, 1)
            mpn = mp.mpf(n)
            mpp = probability_thr

            def binomial_density(tensor: torch.tensor) -> torch.Tensor:
                return binomial(mpn, to_mp(tensor)) * (1 - mpp) ** tensor * mpp ** (mpn - tensor)

            def integral(tensor: torch.Tensor) -> torch.Tensor:
                return integrate.quad(binomial_density, tensor, n)[0]

            integral_array = np.vectorize(integral)
            prob = integral_array(x)
            log_prob = torch.tensor(np.log10(prob))

        return log_prob
