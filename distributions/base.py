from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch


class Energy(ABC):
    """Base class for energy functions E: R^d -> R.

    The Boltzmann target is pi_beta(x) ∝ exp(-beta * E(x)).
    Subclasses must implement energy() and dim.
    grad_energy() defaults to autograd but can be overridden for speed/correctness.
    """

    @abstractmethod
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy for a batch of points.

        Args:
            x: shape [N, d] or [d]

        Returns:
            Scalar energies, shape [N] or []
        """
        ...

    def grad_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient of energy w.r.t. x via autograd.

        Args:
            x: shape [N, d]

        Returns:
            Gradients, shape [N, d]
        """
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            e = self.energy(x)
            grads = torch.autograd.grad(e.sum(), x)[0]
        return grads.detach()

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the input space."""
        ...

    def sample(self, n: int, device: torch.device = torch.device("cpu")) -> Optional[torch.Tensor]:
        """Draw analytic samples if available; otherwise return None."""
        return None


class BoltzmannDistribution:
    """Wraps an Energy to define pi_beta(x) ∝ exp(-beta * E(x)).

    Provides unnormalised log-probability and the corresponding score
    (negative energy gradient scaled by beta).
    """

    def __init__(self, energy: Energy, beta: float):
        self.energy = energy
        self.beta = beta

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Unnormalised log probability: -beta * E(x)."""
        return -self.beta * self.energy.energy(x)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Score of the Boltzmann distribution: -beta * grad E(x)."""
        return -self.beta * self.energy.grad_energy(x)

    def sample(self, n: int, device: torch.device = torch.device("cpu")) -> Optional[torch.Tensor]:
        """Delegate to energy.sample() if available."""
        return self.energy.sample(n, device=device)

    @property
    def dim(self) -> int:
        return self.energy.dim
