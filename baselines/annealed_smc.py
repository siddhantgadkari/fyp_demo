"""Classical annealed SMC baseline using Langevin mutation kernel."""
from __future__ import annotations

from typing import Callable

import torch

from ..fkc.weight_updates import annealing_weight_update
from ..smc.particles import ParticleCloud
from ..smc.smc_sampler import SMCSampler
from .langevin import ULA


class ClassicalAnnealedSMC:
    """Standard annealed importance sampling / SMC with ULA mutation.

    This is the primary baseline against which diffusion-informed SMC is compared.

    Algorithm at each beta step:
        1. Run `n_langevin_steps` of ULA targeting pi_{beta_curr}
        2. Compute incremental weight: log w += -(beta_curr - beta_prev) * E(x_mutated)
        3. Resample if ESS < threshold * N
    """

    def __init__(
        self,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        langevin_step_size: float = 1e-3,
        n_langevin_steps: int = 10,
        ess_threshold: float = 0.5,
        resampling_method: str = "systematic",
    ):
        self.energy_fn = energy_fn
        self.ula = ULA(energy_fn, step_size=langevin_step_size)
        self.n_langevin_steps = n_langevin_steps
        self.ess_threshold = ess_threshold
        self.resampling_method = resampling_method

    def build_sampler(self) -> SMCSampler:
        """Build an SMCSampler wired to ULA mutation and AIS weight update."""

        def mutation_kernel(x: torch.Tensor, beta: float) -> torch.Tensor:
            return self.ula.run(x, beta, self.n_langevin_steps)

        def weight_update(x: torch.Tensor, beta_prev: float, beta_curr: float) -> torch.Tensor:
            return annealing_weight_update(x, beta_prev, beta_curr, self.energy_fn)

        return SMCSampler(
            mutation_kernel=mutation_kernel,
            weight_update=weight_update,
            energy_fn=self.energy_fn,
            ess_threshold=self.ess_threshold,
            resampling_method=self.resampling_method,
        )

    def run(
        self,
        x_init: torch.Tensor,
        beta_ladder: torch.Tensor,
        show_progress: bool = True,
    ):
        """Convenience wrapper: run the full SMC from x_init.

        Args:
            x_init:      [N, d] initial particles (should approximate pi_{beta_ladder[0]})
            beta_ladder: [K+1] inverse temperatures
            show_progress: tqdm progress bar

        Returns:
            (final_cloud, diagnostics)
        """
        N = x_init.shape[0]
        initial_cloud = ParticleCloud(
            x=x_init.clone(),
            log_weights=torch.zeros(N, device=x_init.device),
        )
        sampler = self.build_sampler()
        return sampler.run(initial_cloud, beta_ladder, show_progress=show_progress)
