from __future__ import annotations

import torch

from .base import Energy


class Quadratic(Energy):
    def __init__(self, dim: int = 2):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
            return self._energy(x).squeeze(0)
        return self._energy(x)

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x**2).sum(-1)

    @property
    def global_minima(self) -> torch.Tensor | None:
        return torch.zeros(1, self.dim)

    @property
    def global_minimum_energy(self) -> float | None:
        return 0.0