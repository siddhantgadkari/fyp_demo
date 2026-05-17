from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class DiffusionSchedule(ABC):
    """Base class for a Gaussian corruption schedule.

    The forward process is:
        x_t = alpha(t) * x_0 + sigma(t) * eps,  eps ~ N(0, I)

    t=0 is clean data, t=1 is (approximately) pure noise.
    """

    @abstractmethod
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Signal scaling coefficient at time t. Shape matches t."""
        ...

    @abstractmethod
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Noise standard deviation at time t. Shape matches t."""
        ...

    def marginal_params(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (alpha(t), sigma(t))."""
        return self.alpha(t), self.sigma(t)

    def marginal_sample(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t ~ q(x_t | x_0).

        Args:
            x0: [N, d]
            t:  [N] or scalar

        Returns:
            x_t: [N, d]
            eps: [N, d]  (the noise added, useful for loss computation)
        """
        if t.dim() == 0:
            t = t.expand(x0.shape[0])
        alpha_t = self.alpha(t).view(-1, *([1] * (x0.ndim - 1)))
        sigma_t = self.sigma(t).view(-1, *([1] * (x0.ndim - 1)))
        eps = torch.randn_like(x0)
        x_t = alpha_t * x0 + sigma_t * eps
        return x_t, eps

    def conditional_score(self, x_t: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Analytic score of q(x_t | x_0):
            ∇_{x_t} log q(x_t | x_0) = -(x_t - alpha_t * x_0) / sigma_t^2
        """
        if t.dim() == 0:
            t = t.expand(x_t.shape[0])
        alpha_t = self.alpha(t).view(-1, *([1] * (x_t.ndim - 1)))
        sigma_t = self.sigma(t).view(-1, *([1] * (x_t.ndim - 1)))
        return -(x_t - alpha_t * x0) / (sigma_t**2 + 1e-8)

    def eps_to_score(self, eps_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convert predicted noise to score: score = -eps / sigma(t)."""
        if t.dim() == 0:
            t = t.expand(eps_pred.shape[0])
        sigma_t = self.sigma(t).view(-1, *([1] * (eps_pred.ndim - 1)))
        return -eps_pred / (sigma_t + 1e-8)
