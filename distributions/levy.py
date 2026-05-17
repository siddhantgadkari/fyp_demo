from __future__ import annotations

import math

import torch

from .base import Energy

class Levy(Energy): 
    """Levy function: highly multimodal with many local minima surrounding a 
    single global minimum. More challenging than simple bowl-shaped objectives 
    due to oscillatory structure.

    Global minimum at x=(1,...,1) with E=0.
    Default domain: x_i in [-10, 10].
    """

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
        w = 1 + (x - 1) / 4

        term1 = torch.sin(math.pi * w[:, 0]) ** 2

        wi = w[:, :-1]
        term2 = ((wi - 1) ** 2 * (1 + 10 * torch.sin(math.pi * wi + 1) ** 2)).sum(-1)

        wd = w[:, -1]
        term3 = (wd - 1) ** 2 * (1 + torch.sin(2 * math.pi * wd) ** 2)

        return term1 + term2 + term3
            
        
    @property
    def global_minima(self) -> torch.Tensor | None:
        return torch.ones(1, self.dim) 

    @property
    def global_minimum_energy(self) -> float | None:
        return 0.0 
        