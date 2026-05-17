from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ParticleCloud:
    """A weighted collection of particles.

    Attributes:
        x:           [N, d]  particle positions
        log_weights: [N]     unnormalised log importance weights
        ancestry:    Optional list of ancestry indices from resampling steps
    """

    x: torch.Tensor
    log_weights: torch.Tensor
    ancestry: Optional[list] = field(default=None, repr=False)

    def __post_init__(self):
        assert self.x.ndim == 2
        assert self.log_weights.shape == (self.x.shape[0],)

    @property
    def n_particles(self) -> int:
        return self.x.shape[0]

    @property
    def dim(self) -> int:
        return self.x.shape[1]

    def normalised_weights(self) -> torch.Tensor:
        """Numerically stable softmax of log_weights."""
        lw = self.log_weights - self.log_weights.max()
        w = torch.exp(lw)
        return w / w.sum()

    def ess(self) -> float:
        """Effective Sample Size: (sum w)^2 / sum(w^2)."""
        w = self.normalised_weights()
        return (1.0 / (w**2).sum()).item()

    def ess_ratio(self) -> float:
        """ESS / N."""
        return self.ess() / self.n_particles

    def weighted_mean(self) -> torch.Tensor:
        """Importance-weighted mean of particles."""
        w = self.normalised_weights().unsqueeze(1)
        return (w * self.x).sum(0)

    def clone(self) -> "ParticleCloud":
        return ParticleCloud(
            x=self.x.clone(),
            log_weights=self.log_weights.clone(),
            ancestry=list(self.ancestry) if self.ancestry is not None else None,
        )
