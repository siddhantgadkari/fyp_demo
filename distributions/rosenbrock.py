from __future__ import annotations

import torch

from .base import Energy


class Rosenbrock(Energy):
    """Rosenbrock (banana) function: curved valley, difficult optimisation.

    E(x) = sum_{i=0}^{d-2} [a*(x_{i+1} - x_i^2)^2 + (1-x_i)^2]

    Global minimum at x=(1,...,1) with E=0.
    """

    def __init__(self, dim: int = 2, a: float = 100.0):
        assert dim >= 2
        self._dim = dim
        self.a = a

    @property
    def dim(self) -> int:
        return self._dim

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
            return self._energy(x).squeeze(0)
        return self._energy(x)

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        xi = x[:, :-1]
        xi1 = x[:, 1:]
        return (self.a * (xi1 - xi**2)**2 + (1 - xi)**2).sum(-1)
    
    @property
    def global_minima(self) -> torch.Tensor | None:
        return torch.ones(1, self.dim) 

    @property
    def global_minimum_energy(self) -> float | None:
        return 0.0 

