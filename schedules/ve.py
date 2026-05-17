from __future__ import annotations

import torch

from .base import DiffusionSchedule


class VESchedule(DiffusionSchedule):
    """Variance-Exploding (VE) diffusion schedule.

    sigma(t) = sigma_min * (sigma_max / sigma_min)^t
    alpha(t) = 1  (no signal scaling)

    Forward process:  x_t = x_0 + sigma(t) * eps
    """

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        ratio = self.sigma_max / self.sigma_min
        return self.sigma_min * (ratio**t)

    def reverse_drift(self, t: torch.Tensor, x: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        """Reverse SDE drift for VE: f_rev(t, x) = -d(sigma^2)/dt * score / 2.

        d(sigma^2)/dt = 2 * sigma(t) * dsigma/dt
                       = 2 * sigma(t)^2 * log(sigma_max/sigma_min)
        So f_rev = -sigma(t)^2 * log(sigma_max/sigma_min) * score
        """
        import math
        log_ratio = math.log(self.sigma_max / self.sigma_min)
        sigma_t = self.sigma(t).view(-1, *([1] * (x.ndim - 1)))
        return -(sigma_t**2) * log_ratio * score

    def reverse_diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient for reverse VE SDE."""
        import math
        log_ratio = math.log(self.sigma_max / self.sigma_min)
        sigma_t = self.sigma(t).view(-1, *([1] * (x.ndim - 1)))
        return sigma_t * math.sqrt(2 * log_ratio)
