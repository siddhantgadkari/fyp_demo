from __future__ import annotations

import torch

from .base import Energy


class DoubleWell(Energy):
    """Double-well potential: E(x) = a*(x[0]^4 - b*x[0]^2) + c*sum(x[1:]^2).

    Minima near x[0] ≈ ±sqrt(b/2) for the first coordinate;
    remaining coordinates are harmonic.
    """

    def __init__(self, dim: int = 2, a: float = 1.0, b: float = 4.0, c: float = 1.0):
        self._dim = dim
        self.a = a
        self.b = b
        self.c = c

    @property
    def dim(self) -> int:
        return self._dim

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
            return self._energy(x).squeeze(0)
        return self._energy(x)

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, 0]
        well = self.a * (x0**4 - self.b * x0**2)
        if self._dim > 1:
            harmonic = self.c * (x[:, 1:] ** 2).sum(-1)
        else:
            harmonic = torch.zeros_like(well)
        return well + harmonic


class ManyWell(Energy):
    """Many-well potential: product of double-wells in pairs of dimensions.

    E(x) = sum_{k} a*(x[2k]^4 - b*x[2k]^2) + c*x[2k+1]^2

    For odd-dimensional inputs the last coordinate gets a harmonic term.
    """

    def __init__(self, dim: int = 4, a: float = 1.0, b: float = 4.0, c: float = 1.0):
        assert dim >= 2
        self._dim = dim
        self.a = a
        self.b = b
        self.c = c

    @property
    def dim(self) -> int:
        return self._dim

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
            return self._energy(x).squeeze(0)
        return self._energy(x)

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        k = 0
        while 2 * k + 1 < self._dim:
            x0 = x[:, 2 * k]
            x1 = x[:, 2 * k + 1]
            total += self.a * (x0**4 - self.b * x0**2) + self.c * x1**2
            k += 1
        if self._dim % 2 == 1:
            total += self.c * x[:, -1] ** 2
        return total
