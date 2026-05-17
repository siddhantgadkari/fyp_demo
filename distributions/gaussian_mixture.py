from __future__ import annotations

import math
from typing import Optional

import torch
import torch.distributions as D

from .base import Energy


class GaussianMixture(Energy):
    """Isotropic Gaussian mixture energy: E(x) = -log p_GMM(x).

    At inverse temperature beta, the Boltzmann target is:
        pi_beta(x) ∝ exp(-beta * E(x)) = p_GMM(x)^beta

    For analytic samples at beta=1 we can draw from the mixture directly.
    For beta != 1 we return None from sample() (use MCMC instead).
    """

    def __init__(
        self,
        means: torch.Tensor,
        stds: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            means:   [K, d] component means
            stds:    [K] or [K, d] component standard deviations (isotropic if [K])
            weights: [K] mixture weights (uniform if None)
        """
        assert means.ndim == 2
        K, d = means.shape
        self._dim = d

        if stds.ndim == 1:
            stds = stds.unsqueeze(1).expand(K, d)
        assert stds.shape == (K, d)

        if weights is None:
            weights = torch.ones(K) / K
        weights = weights / weights.sum()

        self.register_params(means, stds, weights)

    def register_params(self, means, stds, weights):
        self.means = means
        self.stds = stds
        self.weights = weights

    @classmethod
    def random_2d(
        cls,
        n_components: int = 8,
        radius: float = 5.0,
        std: float = 0.5,
        seed: int = 0,
    ) -> "GaussianMixture":
        """Create a 2D mixture with components arranged on a ring."""
        torch.manual_seed(seed)
        angles = torch.linspace(0, 2 * math.pi, n_components + 1)[:-1]
        means = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
        stds = torch.full((n_components,), std)
        return cls(means, stds)

    @classmethod
    def double_ring_2d(
        cls,
        n_inner: int = 6,
        n_outer: int = 10,
        r_inner: float = 3.0,
        r_outer: float = 7.0,
        std: float = 0.5,
        seed: int = 0,
    ) -> "GaussianMixture":
        """Two concentric rings of Gaussians."""
        torch.manual_seed(seed)
        angles_i = torch.linspace(0, 2 * math.pi, n_inner + 1)[:-1]
        angles_o = torch.linspace(0, 2 * math.pi, n_outer + 1)[:-1]
        means_i = torch.stack([r_inner * torch.cos(angles_i), r_inner * torch.sin(angles_i)], dim=1)
        means_o = torch.stack([r_outer * torch.cos(angles_o), r_outer * torch.sin(angles_o)], dim=1)
        means = torch.cat([means_i, means_o], dim=0)
        stds = torch.full((n_inner + n_outer,), std)
        return cls(means, stds)

    @property
    def dim(self) -> int:
        return self._dim

    def _log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mixture log-probability."""
        means = self.means.to(x.device)
        stds = self.stds.to(x.device)
        weights = self.weights.to(x.device)

        # x: [N, d], means: [K, d]
        diff = x.unsqueeze(1) - means.unsqueeze(0)  # [N, K, d]
        log_var = 2 * torch.log(stds).unsqueeze(0)  # [1, K, d]
        log_gauss = -0.5 * (diff**2 / stds.unsqueeze(0)**2 + log_var + math.log(2 * math.pi))
        log_gauss = log_gauss.sum(-1)  # [N, K]
        log_w = torch.log(weights).unsqueeze(0)  # [1, K]
        return torch.logsumexp(log_gauss + log_w, dim=1)  # [N]

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
            return -self._log_prob(x).squeeze(0)
        return -self._log_prob(x)

    def grad_energy(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().requires_grad_(True)
        e = self.energy(x)
        return torch.autograd.grad(e.sum(), x)[0].detach()

    def sample(self, n: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Exact samples from the mixture at beta=1."""
        means = self.means.to(device)
        stds = self.stds.to(device)
        weights = self.weights.to(device)

        K = means.shape[0]
        component_idx = torch.multinomial(weights, n, replacement=True)
        chosen_means = means[component_idx]  # [N, d]
        chosen_stds = stds[component_idx]    # [N, d]
        eps = torch.randn_like(chosen_means)
        return chosen_means + chosen_stds * eps
